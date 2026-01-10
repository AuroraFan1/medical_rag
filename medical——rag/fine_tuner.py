"""
模型微调模块
针对医疗文本优化
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_dataset
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import gc

from config import (
    MODEL_CONFIGS, FINETUNE_CONFIG, MODELS_DIR,
    DEFAULT_LLM, DEVICE
)

import os
import torch

# 设置内存优化环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 清理GPU缓存
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    print(f"初始GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

logger = logging.getLogger(__name__)

class MedicalModelFinetuner:
    """医疗模型微调器"""

    def __init__(self,
                 base_model: str = DEFAULT_LLM,
                 output_dir: Optional[str] = None):

        self.base_model = base_model
        self.model_config = MODEL_CONFIGS.get(base_model, MODEL_CONFIGS[DEFAULT_LLM])

        # 输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = MODELS_DIR / f"finetuned_{base_model.replace('.', '_')}"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 模型组件
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.trainer = None

        # 训练状态
        self.training_state = {
            "status": "not_started",
            "epoch": 0,
            "step": 0,
            "loss": None
        }

    def _setup_model_and_tokenizer(self):
        """设置模型和tokenizer"""
        logger.info(f"加载基础模型: {self.base_model}")

        try:
            # 检查本地模型路径
            local_path = self.model_config.get("local_path")
            if local_path and Path(local_path).exists():
                model_path = local_path
                logger.info(f"使用本地模型: {model_path}")
            else:
                model_path = self.model_config["name"]
                logger.info(f"从HuggingFace下载模型: {model_path}")

            # 量化配置
            bnb_config = None
            if FINETUNE_CONFIG['quantization']['load_in_4bit']:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=str(MODELS_DIR / "llm_models")
            )

            # 设置特殊token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 打印内存状态
            if torch.cuda.is_available():
                print(f"加载模型前GPU内存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(MODELS_DIR / "llm_models"),
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_cache=False,  # 禁用缓存
                max_memory={0: "24GB"},  # 限制GPU0使用30GB
                #attn_implementation="flash_attention_2",
                #offload_floder="./offload",
            )

            self.model.gradient_checkpointing_enable()

            # 准备k-bit训练
            if bnb_config:
                self.model = prepare_model_for_kbit_training(self.model)

            logger.info("模型和tokenizer加载成功")

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def _setup_lora(self):
        """设置LoRA配置"""
        lora_config = FINETUNE_CONFIG['lora']

        logger.info(f"配置LoRA: r={lora_config['r']}, alpha={lora_config['lora_alpha']}")

        self.peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules'],
            bias=lora_config['bias'],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )

        self.model = get_peft_model(self.model, self.peft_config)

        # 打印可训练参数
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"全部参数: {all_params:,}")
        logger.info(f"可训练参数占比: {100 * trainable_params / all_params:.2f}%")

    def load_training_data(self,
                          train_path: str,
                          val_path: str) -> Dict[str, Dataset]:
        """加载训练数据"""

        def preprocess_function(examples):
            """预处理函数"""
            # 构建输入文本
            inputs = []
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                input_text = examples.get('input', [''] * len(examples['instruction']))[i]
                output = examples['output'][i]

                if input_text:
                    text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n{output}"
                else:
                    text = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"

                inputs.append(text)

            # 编码
            model_inputs = self.tokenizer(
                inputs,
                max_length=FINETUNE_CONFIG['training']['max_seq_length'],
                truncation=True,
                padding=False
            )

            # 设置标签
            labels = model_inputs["input_ids"].copy()
            model_inputs["labels"] = labels

            return model_inputs

        logger.info(f"加载训练数据: {train_path}")
        logger.info(f"加载验证数据: {val_path}")

        # 加载数据集
        try:
            # 尝试从JSONL文件加载
            train_dataset = Dataset.from_json(train_path)
            val_dataset = Dataset.from_json(val_path)

            # 限制样本数（用于测试）
            max_train_samples = FINETUNE_CONFIG['data']['max_train_samples']
            max_val_samples = FINETUNE_CONFIG['data']['max_eval_samples']

            if len(train_dataset) > max_train_samples:
                train_dataset = train_dataset.select(range(max_train_samples))

            if len(val_dataset) > max_val_samples:
                val_dataset = val_dataset.select(range(max_val_samples))

            # 预处理
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=4,
                remove_columns=train_dataset.column_names,
                desc="预处理训练数据"
            )

            val_dataset = val_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=4,
                remove_columns=val_dataset.column_names,
                desc="预处理验证数据"
            )

            logger.info(f"训练集大小: {len(train_dataset)}")
            logger.info(f"验证集大小: {len(val_dataset)}")

            return {
                "train": train_dataset,
                "validation": val_dataset
            }

        except Exception as e:
            logger.error(f"加载训练数据失败: {e}")
            raise

    def _compute_metrics(self, eval_pred):
        """计算评估指标"""
        import numpy as np

        predictions, labels = eval_pred

        # 解码预测（取argmax）
        predictions = np.argmax(predictions, axis=-1)

        # 计算准确率
        correct = (predictions == labels).sum()
        total = labels.size

        accuracy = correct / total if total > 0 else 0

        return {"accuracy": accuracy}

    def setup_trainer(self, datasets: Dict[str, Dataset]):
        """设置Trainer"""

        training_config = FINETUNE_CONFIG['training']

        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=training_config['num_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=training_config['logging_steps'],
            eval_steps=training_config['eval_steps'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            max_grad_norm=training_config['max_grad_norm'],
            eval_strategy="steps",
            save_strategy="steps",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=training_config.get('fp16', True),
            bf16=training_config.get('bf16', False),
            push_to_hub=False,
            report_to="tensorboard",
            dataloader_num_workers=4,
            remove_unused_columns=False,
            group_by_length=True,
            length_column_name="length",
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
        )

        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )

        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics
        )

        logger.info("Trainer设置完成")

    def train(self,
              train_path: str,
              val_path: str,
              resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """执行训练"""

        logger.info("开始模型微调训练")

        try:

            gc.collect()
            torch.cuda.empty_cache()

            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
                print(f"初始GPU内存占用{initial_memory / 1024**3:.2f} GB")

            # 1. 设置模型和tokenizer
            self._setup_model_and_tokenizer()

            # 2. 设置LoRA
            self._setup_lora()

            # 3. 加载数据
            datasets = self.load_training_data(train_path, val_path)

            # 4. 设置Trainer
            self.setup_trainer(datasets)

            # 5. 训练
            self.training_state["status"] = "training"

            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # 6. 保存模型
            self.trainer.save_model()
            self.tokenizer.save_pretrained(str(self.output_dir))

            # 7. 保存训练指标
            metrics = train_result.metrics
            metrics_file = self.output_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            # 更新训练状态
            self.training_state.update({
                "status": "completed",
                "epoch": metrics.get("epoch", 0),
                "step": metrics.get("step", 0),
                "loss": metrics.get("train_loss", None),
                "eval_loss": metrics.get("eval_loss", None)
            })

            logger.info(f"训练完成！模型保存到: {self.output_dir}")

            return {
                "output_dir": str(self.output_dir),
                "metrics": metrics,
                "training_state": self.training_state
            }

        except Exception as e:
            logger.error(f"训练失败: {e}")
            self.training_state["status"] = "failed"
            raise

    def evaluate(self, test_path: Optional[str] = None):
        """评估模型"""
        if not self.trainer:
            logger.error("Trainer未初始化")
            return None

        try:
            if test_path:
                # 加载测试数据
                test_dataset = Dataset.from_json(test_path)
                test_dataset = test_dataset.map(
                    lambda examples: self.tokenizer(
                        examples["instruction"],
                        max_length=FINETUNE_CONFIG['training']['max_seq_length'],
                        truncation=True,
                        padding=True
                    ),
                    batched=True
                )

                # 评估
                metrics = self.trainer.evaluate(eval_dataset=test_dataset)
            else:
                # 使用验证集评估
                metrics = self.trainer.evaluate()

            logger.info(f"评估结果: {metrics}")

            # 保存评估结果
            eval_file = self.output_dir / "evaluation_metrics.json"
            with open(eval_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            return metrics

        except Exception as e:
            logger.error(f"评估失败: {e}")
            return None

    def merge_and_save_full_model(self, output_path: Optional[str] = None):
        """合并LoRA权重并保存完整模型"""
        if self.model is None:
            raise ValueError("模型未加载")

        try:
            # 合并LoRA权重
            logger.info("合并LoRA权重...")
            merged_model = self.model.merge_and_unload()

            # 保存路径
            if output_path:
                save_path = Path(output_path)
            else:
                save_path = self.output_dir / "merged_full_model"

            save_path.mkdir(parents=True, exist_ok=True)

            # 保存完整模型
            merged_model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))

            logger.info(f"完整模型已保存到: {save_path}")

            # 清理内存
            del merged_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return str(save_path)

        except Exception as e:
            logger.error(f"合并模型失败: {e}")
            raise

    def get_training_info(self) -> Dict[str, Any]:
        """获取训练信息"""
        info = {
            "base_model": self.base_model,
            "output_dir": str(self.output_dir),
            "training_state": self.training_state,
            "peft_config": self.peft_config.__dict__ if self.peft_config else None,
            "has_model": self.model is not None,
            "has_tokenizer": self.tokenizer is not None
        }

        return info

def run_finetuning_pipeline(
    train_path: str,
    val_path: str,
    base_model: str = DEFAULT_LLM,
    output_dir: Optional[str] = None,
    test_mode: bool = False,
    merge_model: bool = True
) -> Dict[str, Any]:
    """
    运行微调管道

    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径
        base_model: 基础模型
        output_dir: 输出目录
        test_mode: 测试模式
        merge_model: 是否合并模型

    Returns:
        训练结果
    """

    logger.info("="*60)
    logger.info("运行模型微调管道")
    logger.info("="*60)

    # 检查数据文件
    train_path_obj = Path(train_path)
    val_path_obj = Path(val_path)

    if not train_path_obj.exists():
        logger.error(f"训练文件不存在: {train_path}")
        return {}

    if not val_path_obj.exists():
        logger.error(f"验证文件不存在: {val_path}")
        return {}

    # 创建微调器
    finetuner = MedicalModelFinetuner(
        base_model=base_model,
        output_dir=output_dir
    )

    # 测试模式调整配置
    if test_mode:
        logger.info("测试模式：使用小规模配置")
        # 可以在这里调整FINETUNE_CONFIG用于测试

    # 执行微调
    try:
        result = finetuner.train(train_path, val_path)

        # 合并模型（可选）
        if merge_model:
            merged_path = finetuner.merge_and_save_full_model()
            result["merged_model_path"] = merged_path

        # 评估模型
        eval_result = finetuner.evaluate(val_path)
        if eval_result:
            result["evaluation"] = eval_result

        logger.info("✅ 模型微调完成")

        # 打印结果摘要
        print("\n" + "="*60)
        print("📊 微调结果摘要")
        print("="*60)
        print(f"基础模型: {base_model}")
        print(f"输出目录: {result['output_dir']}")
        print(f"训练损失: {result['metrics'].get('train_loss', 'N/A')}")
        print(f"评估损失: {result['metrics'].get('eval_loss', 'N/A')}")

        if "merged_model_path" in result:
            print(f"完整模型: {result['merged_model_path']}")

        print("="*60)

        return result

    except Exception as e:
        logger.error(f"微调失败: {e}")
        return {}

def test_finetuning():
    """测试微调功能"""

    print("测试模型微调功能...")

    # 创建测试数据
    test_data = [
        {
            "instruction": "糖尿病患者应该注意什么饮食？",
            "input": "患者有2型糖尿病，血糖控制不佳",
            "output": "建议低糖低脂饮食，控制碳水化合物摄入，多吃蔬菜和全谷物，定期监测血糖。"
        },
        {
            "instruction": "高血压怎么治疗？",
            "input": "血压150/95mmHg，无其他疾病",
            "output": "建议低盐饮食，适度运动，如血压持续偏高需在医生指导下服用降压药物。"
        }
    ]

    # 保存测试数据
    test_dir = MODELS_DIR / "test_finetune_data"
    test_dir.mkdir(exist_ok=True)

    train_path = test_dir / "train.jsonl"
    val_path = test_dir / "val.jsonl"

    with open(train_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(val_path, 'w', encoding='utf-8') as f:
        for item in test_data[:1]:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"创建测试数据: {train_path}, {val_path}")

    # 运行微调（测试模式）
    result = run_finetuning_pipeline(
        train_path=str(train_path),
        val_path=str(val_path),
        base_model=DEFAULT_LLM,
        test_mode=True,
        merge_model=False
    )

    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗模型微调")
    parser.add_argument("--train-path", type=str, required=True, help="训练数据路径")
    parser.add_argument("--val-path", type=str, required=True, help="验证数据路径")
    parser.add_argument("--base-model", type=str, default=DEFAULT_LLM, help="基础模型")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--merge", action="store_true", help="合并完整模型")

    args = parser.parse_args()

    if args.test:
        # 测试模式
        test_finetuning()
    else:
        # 正式微调
        result = run_finetuning_pipeline(
            train_path=args.train_path,
            val_path=args.val_path,
            base_model=args.base_model,
            output_dir=args.output_dir,
            merge_model=args.merge
        )

        if result:
            print(f"\n微调完成，结果保存在: {result['output_dir']}")
        else:
            print("微调失败")