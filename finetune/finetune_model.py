"""
模型微调模块
使用LoRA等技术对医疗模型进行微调
"""

import os
import sys
import torch
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer
import datasets
import pandas as pd
import json

from config import MODEL_NAME

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """微调配置"""
    # 模型配置
    base_model: str = MODEL_NAME
    output_dir: str = "./models/finetuned_medical"

    # 数据配置
    train_file: str = "./finetune_data/train.json"
    val_file: str = "./finetune_data/val.json"

    # 训练参数
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500

    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # 量化配置
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # 其他配置
    max_seq_length: int = 1024
    dataset_text_field: str = "text"
    packing: bool = False
    fp16: bool = True
    bf16: bool = False


class MedicalModelFinetuner:
    """医疗模型微调器"""

    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """设置模型和tokenizer"""
        logger.info(f"加载基础模型: {self.config.base_model}")
        import os
        from pathlib import Path

        # 1. 首先检查是否有本地模型路径
        local_model_path = "/root/autodl-tmp/study/models1/qwen/Qwen2.5-7B-Instruct"  # 您手动下载的路径
        if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
            logger.info(f"找到本地模型: {local_model_path}")
            model_path = local_model_path
            print(model_path)

        # 量化配置
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.use_4bit,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="cuda:0",
            trust_remote_code=True,
            use_cache=False
        )

        # 准备k-bit训练
        self.model = prepare_model_for_kbit_training(self.model)

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("模型和tokenizer加载完成")

    def setup_lora(self):
        """设置LoRA配置"""
        logger.info("配置LoRA...")

        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self):
        """准备数据集"""
        logger.info(f"加载训练数据: {self.config.train_file}")
        logger.info(f"加载验证数据: {self.config.val_file}")

        def format_instruction(example):
            """格式化指令"""
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')

            if input_text:
                text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n{output}"
            else:
                text = f"### 指令:\n{instruction}\n\n### 回答:\n{output}"

            return {"text": text}

        # 加载数据集
        train_dataset = datasets.Dataset.from_json(self.config.train_file)
        val_dataset = datasets.Dataset.from_json(self.config.val_file)

        # 格式化数据
        train_dataset = train_dataset.map(format_instruction)
        val_dataset = val_dataset.map(format_instruction)

        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")

        return train_dataset, val_dataset

    def setup_trainer(self, train_dataset, eval_dataset):
        """设置训练器"""
        logger.info("设置训练器...")

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard",
            push_to_hub=False,
        )

        # 创建训练器
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=None,  # 已经在模型中配置了
            args=training_args,
            processing_class=None,
            data_collator=None,
        )

    def train(self):
        """开始训练"""
        logger.info("开始微调训练...")

        # 设置模型
        self.setup_model()

        # 设置LoRA
        self.setup_lora()

        # 准备数据集
        train_dataset, eval_dataset = self.prepare_dataset()

        # 设置训练器
        self.setup_trainer(train_dataset, eval_dataset)

        # 训练模型
        train_result = self.trainer.train()

        # 保存模型
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        # 保存训练指标
        metrics = train_result.metrics
        metrics_file = os.path.join(self.config.output_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"训练完成！模型保存到: {self.config.output_dir}")

        return metrics

    def evaluate(self, eval_dataset=None):
        """评估模型"""
        if self.trainer is None:
            logger.error("训练器未初始化")
            return None

        logger.info("评估模型...")

        if eval_dataset is None:
            _, eval_dataset = self.prepare_dataset()

        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)

        logger.info(f"评估结果: {metrics}")

        # 保存评估结果
        metrics_file = os.path.join(self.config.output_dir, "evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics


def run_finetune(config_dict: Dict[str, Any] = None):
    """运行微调的主函数"""
    config = FinetuneConfig()

    # 更新配置
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # 检查数据文件
    if not os.path.exists(config.train_file):
        logger.error(f"训练文件不存在: {config.train_file}")
        logger.info("请先运行 finetune_data.py 准备数据")
        return None

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 保存配置
    config_file = os.path.join(config.output_dir, "finetune_config.json")
    with open(config_file, 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    # 运行微调
    finetuner = MedicalModelFinetuner(config)
    metrics = finetuner.train()

    return {
        "output_dir": config.output_dir,
        "metrics": metrics,
        "config": config.__dict__
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗模型微调")
    parser.add_argument("--train-file", type=str, default="./finetune_data/train.json")
    parser.add_argument("--val-file", type=str, default="./finetune_data/val.json")
    parser.add_argument("--output-dir", type=str, default="./models/finetuned_medical")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--test", action="store_true", help="测试模式")

    args = parser.parse_args()

    # 测试模式使用小数据集
    if args.test:
        config_dict = {
            "train_file": args.train_file,
            "val_file": args.val_file,
            "output_dir": args.output_dir,
            "num_epochs": 1,
            "per_device_train_batch_size": 2,
            "logging_steps": 5,
            "eval_steps": 10,
            "save_steps": 20
        }
    else:
        config_dict = {
            "train_file": args.train_file,
            "val_file": args.val_file,
            "output_dir": args.output_dir,
            "num_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "learning_rate": args.learning_rate
        }

    result = run_finetune(config_dict)

    if result:
        print("\n" + "=" * 50)
        print("✅ 模型微调完成")
        print("=" * 50)
        print(f"模型保存位置: {result['output_dir']}")
        if result.get('metrics'):
            print(f"训练损失: {result['metrics'].get('train_loss', 'N/A')}")
            print(f"评估损失: {result['metrics'].get('eval_loss', 'N/A')}")