"""
LLM管理器
支持本地模型和API，针对医疗文本优化
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer
)
from typing import List, Dict, Any, Optional, Generator
import logging
from openai import OpenAI
import backoff
import json
from pathlib import Path

from config import (
    MODEL_CONFIGS, DEFAULT_LLM, SYSTEM_CONFIG,
    OPENAI_CONFIG, DEVICE, MODELS_DIR, PROMPT_TEMPLATES
)

logger = logging.getLogger(__name__)

class StopOnTokens(StoppingCriteria):
    """自定义停止条件"""
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class LLMManager:
    """LLM管理器"""

    def __init__(self,
                 model_name: str = DEFAULT_LLM,
                 use_api: bool = False,
                 api_config: Optional[Dict] = None,
                 load_in_4bit: bool = True):

        self.model_name = model_name
        self.use_api = use_api
        self.api_config = api_config or OPENAI_CONFIG
        self.load_in_4bit = load_in_4bit

        if not use_api:
            self._load_local_model()
        else:
            self._init_api_client()

    def _load_local_model(self):
        """加载本地模型"""
        # 首先检查是否是本地路径
        local_path = Path(self.model_name)
        if local_path.exists():
            # 是本地路径，直接使用
            model_path = str(local_path)
            logger.info(f"加载本地模型路径: {model_path}")

            # 创建默认配置，因为我们不知道具体参数
            config = {
                'name': self.model_name,
                'quantization': '4bit' if self.load_in_4bit else 'fp16',
                'max_length': 4096  # 默认值
            }

        elif self.model_name in MODEL_CONFIGS:
            config = MODEL_CONFIGS[self.model_name]
            logger.info(f"加载已知模型: {config['name']}")

            # 检查本地是否有模型
            local_path = config.get("local_path")
            if local_path and Path(local_path).exists():
                model_path = local_path
                logger.info(f"使用本地模型: {model_path}")
            else:
                model_path = config["name"]
                logger.info(f"从HuggingFace下载模型: {model_path}")
        else:
            logger.warning(f"未知模型 {self.model_name}，使用默认模型")
            self.model_name = DEFAULT_LLM
            config = MODEL_CONFIGS[self.model_name]

            local_path = config.get("local_path")
            if local_path and Path(local_path).exists():
                model_path = local_path
            else:
                model_path = config["name"]

        try:
            # 量化配置
            quantization_config = None
            if self.load_in_4bit and config.get('quantization') == '4bit':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_storage=torch.float16
                )

            # 加载tokenizer - 添加更多容错选项
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    cache_dir=str(MODELS_DIR / "llm_models") if MODELS_DIR else None
                )
            except Exception as tokenizer_error:
                logger.warning(f"使用默认tokenizer加载失败: {tokenizer_error}")
                # 尝试使用基础模型的tokenizer
                default_config = MODEL_CONFIGS.get(DEFAULT_LLM, {})
                default_path = default_config.get("local_path",
                                                  default_config.get("name", "Qwen/Qwen2.5-1.5B-Instruct"))
                self.tokenizer = AutoTokenizer.from_pretrained(
                    default_path,
                    trust_remote_code=True,
                    cache_dir=str(MODELS_DIR / "llm_models") if MODELS_DIR else None
                )

            # 设置padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型
            torch_dtype = torch.float16 if config.get('quantization') == 'fp16' else None

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(MODELS_DIR / "llm_models") if MODELS_DIR else None,
                max_memory={0: "28GB"} if torch.cuda.is_available() else None
            )

            # 设置生成配置
            self.generation_config = GenerationConfig(
                max_new_tokens=SYSTEM_CONFIG['max_tokens_per_response'],
                temperature=SYSTEM_CONFIG['temperature'],
                top_p=SYSTEM_CONFIG['top_p'],
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=SYSTEM_CONFIG.get('repetition_penalty', 1.1)
            )

            # 停止条件
            self.stopping_criteria = StoppingCriteriaList([
                StopOnTokens([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])
            ])

            logger.info(f"模型加载成功: {model_path}")

            # 更新模型名称显示
            self.model_name = model_path

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    # def _load_local_model(self):
    #     """加载本地模型"""
    #     if self.model_name not in MODEL_CONFIGS:
    #         logger.warning(f"未知模型 {self.model_name}，使用默认模型")
    #         self.model_name = DEFAULT_LLM
    #
    #     config = MODEL_CONFIGS[self.model_name]
    #     logger.info(f"加载本地模型: {config['name']}")
    #
    #     try:
    #         # 检查本地是否有模型
    #         local_path = config.get("local_path")
    #         if local_path and Path(local_path).exists():
    #             model_path = local_path
    #             logger.info(f"使用本地模型: {model_path}")
    #         else:
    #             model_path = config["name"]
    #             logger.info(f"从HuggingFace下载模型: {model_path}")
    #
    #         # 量化配置
    #         quantization_config = None
    #         if self.load_in_4bit and config['quantization'] == '4bit':
    #             quantization_config = BitsAndBytesConfig(
    #                 load_in_4bit=True,
    #                 bnb_4bit_compute_dtype=torch.float16,
    #                 bnb_4bit_quant_type="nf4",
    #                 bnb_4bit_use_double_quant=True,
    #                 bnb_4bit_quant_storage=torch.float16
    #             )
    #
    #         # 加载tokenizer
    #         self.tokenizer = AutoTokenizer.from_pretrained(
    #             model_path,
    #             trust_remote_code=True,
    #             cache_dir=str(MODELS_DIR / "llm_models")
    #         )
    #
    #         # 设置padding token
    #         if self.tokenizer.pad_token is None:
    #             self.tokenizer.pad_token = self.tokenizer.eos_token
    #
    #         # 加载模型
    #         torch_dtype = torch.float16 if config['quantization'] == 'fp16' else None
    #
    #         self.model = AutoModelForCausalLM.from_pretrained(
    #             model_path,
    #             quantization_config=quantization_config,
    #             torch_dtype=torch_dtype,
    #             device_map="auto",
    #             trust_remote_code=True,
    #             cache_dir=str(MODELS_DIR / "llm_models"),
    #             max_memory={0: "28GB"} if torch.cuda.is_available() else None
    #         )
    #
    #         # 设置生成配置
    #         self.generation_config = GenerationConfig(
    #             max_new_tokens=SYSTEM_CONFIG['max_tokens_per_response'],
    #             temperature=SYSTEM_CONFIG['temperature'],
    #             top_p=SYSTEM_CONFIG['top_p'],
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id,
    #             repetition_penalty=SYSTEM_CONFIG.get('repetition_penalty', 1.1)
    #         )
    #
    #         # 停止条件
    #         self.stopping_criteria = StoppingCriteriaList([
    #             StopOnTokens([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])
    #         ])
    #
    #         logger.info(f"本地模型加载成功: {self.model_name}")
    #
    #     except Exception as e:
    #         logger.error(f"加载本地模型失败: {e}")
    #         raise

    def _init_api_client(self):
        """初始化API客户端"""
        try:
            self.client = OpenAI(
                api_key=self.api_config['api_key'],
                base_url=self.api_config['base_url']
            )
            logger.info("API客户端初始化成功")
        except Exception as e:
            logger.error(f"初始化API客户端失败: {e}")
            raise

    def generate(self,
                 prompt: str,
                 context: Optional[str] = None,
                 history: Optional[List[Dict]] = None,
                 streaming: bool = False,
                 max_tokens: Optional[int] = None) -> Generator[str, None, str]:
        """
        生成回答

        Args:
            prompt: 用户问题
            context: 检索上下文
            history: 对话历史
            streaming: 是否流式生成
            max_tokens: 最大token数

        Returns:
            生成的回答
        """

        # 构建完整prompt
        full_prompt = self._build_prompt(prompt, context, history)

        if self.use_api:
            return self._generate_api(full_prompt, streaming, max_tokens)
        else:
            return self._generate_local(full_prompt, streaming, max_tokens)

    def _build_prompt(self,
                     prompt: str,
                     context: Optional[str] = None,
                     history: Optional[List[Dict]] = None) -> str:
        """构建完整prompt"""

        if history and len(history) > 0:
            # 多轮对话
            history_text = ""
            for h in history[-3:]:  # 只保留最近3轮
                if h.get("role") == "user":
                    history_text += f"用户：{h.get('content', '')}\n"
                elif h.get("role") == "assistant":
                    history_text += f"助手：{h.get('content', '')}\n"

            template = PROMPT_TEMPLATES["multi_turn"]
            return template.format(
                history=history_text,
                context=context or "无相关病例信息",
                question=prompt
            )
        else:
            # 单轮对话
            template = PROMPT_TEMPLATES["base"]
            return template.format(
                context=context or "无相关病例信息",
                question=prompt
            )

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _generate_api(self,
                     prompt: str,
                     streaming: bool = False,
                     max_tokens: Optional[int] = None) -> Generator[str, None, str]:
        """使用API生成"""

        try:
            generation_params = {
                "model": self.api_config['model'],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or SYSTEM_CONFIG['max_tokens_per_response'],
                "temperature": SYSTEM_CONFIG['temperature'],
                "top_p": SYSTEM_CONFIG['top_p']
            }

            if streaming:
                response = self.client.chat.completions.create(
                    **generation_params,
                    stream=True
                )

                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content

                return full_response

            else:
                response = self.client.chat.completions.create(**generation_params)
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"API生成失败: {e}")
            return f"生成失败: {str(e)}"

    def _generate_local(self,
                       prompt: str,
                       streaming: bool = False,
                       max_tokens: Optional[int] = None) -> Generator[str, None, str]:
        """使用本地模型生成"""

        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MODEL_CONFIGS[self.model_name]['max_length'] // 2
            ).to(self.model.device)

            # 更新生成配置
            generation_config = self.generation_config.copy()
            if max_tokens:
                generation_config.max_new_tokens = max_tokens

            if streaming:
                # 流式生成
                streamer = TextStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )

                # 生成
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    stopping_criteria=self.stopping_criteria,
                    streamer=streamer,
                    do_sample=True
                )

                # 解码完整响应
                full_response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                # 由于streamer已经输出，这里返回完整响应
                return full_response

            else:
                # 一次性生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                        stopping_criteria=self.stopping_criteria,
                        do_sample=True
                    )

                # 解码响应
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                return response

        except Exception as e:
            logger.error(f"本地生成失败: {e}")
            return f"生成失败: {str(e)}"

    def check_uncertainty(self, response: str, context: str) -> bool:
        """检查回答的不确定性"""

        uncertainty_keywords = [
            "不确定", "无法确定", "不知道", "不清楚",
            "建议咨询", "需要进一步", "无法回答",
            "信息不足", "根据现有信息", "不能确定",
            "请咨询医生", "建议就医", "需要检查"
        ]

        # 检查是否包含不确定性关键词
        response_lower = response.lower()
        for keyword in uncertainty_keywords:
            if keyword in response_lower:
                return True

        # 检查回答是否过短
        if len(response.strip()) < 50:
            return True

        # 检查是否包含具体的医疗建议
        medical_action_keywords = ["建议", "应该", "可以", "需要", "注意"]
        has_medical_advice = any(keyword in response for keyword in medical_action_keywords)

        if not has_medical_advice:
            return True

        return False

    def format_with_citations(self, response: str, sources: List[Dict]) -> str:
        """格式化回答并添加引用"""

        if not sources:
            return response

        # 在回答中添加引用标记
        citation_markers = []

        for i, source in enumerate(sources[:5]):  # 最多引用5个来源
            marker = f"[{i+1}]"
            citation_markers.append(marker)

            # 简单的引用插入（可根据需要增强）
            # 这里可以根据内容相似度在回答中插入引用

        # 在回答末尾添加引用说明
        if citation_markers:
            response += f"\n\n**参考病例来源**: {', '.join(citation_markers)}"

            # 添加来源简要信息
            response += "\n\n**来源信息**:"
            for i, source in enumerate(sources[:3]):  # 显示前3个来源
                disease = source.get("metadata", {}).get("disease", "未知疾病")
                hospital = source.get("metadata", {}).get("hospital", "未知医院")
                response += f"\n[{i+1}] {disease} - {hospital}"

        return response

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "model_name": self.model_name,
            "use_api": self.use_api,
            "config": MODEL_CONFIGS.get(self.model_name, {}),
            "generation_config": {
                "max_tokens": SYSTEM_CONFIG['max_tokens_per_response'],
                "temperature": SYSTEM_CONFIG['temperature'],
                "top_p": SYSTEM_CONFIG['top_p']
            }
        }

        if not self.use_api:
            info.update({
                "device": str(self.model.device),
                "dtype": str(self.model.dtype),
                "parameters": sum(p.numel() for p in self.model.parameters())
            })

        return info

def test_llm():
    """测试LLM功能"""

    print("测试LLM管理器...")

    # 测试本地模型
    try:
        llm = LLMManager(
            model_name="qwen2.5-7b",
            use_api=False,
            load_in_4bit=True
        )

        print("本地模型加载成功")

        # 测试生成
        prompt = "糖尿病患者应该注意什么饮食？"
        response = llm.generate(prompt, streaming=False)

        print(f"\n测试生成:")
        print(f"问题: {prompt}")
        print(f"回答: {response[:200]}...")

        # 模型信息
        info = llm.get_model_info()
        print(f"\n模型信息:")
        for key, value in info.items():
            if key != "config":
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"本地模型测试失败: {e}")

    print("\n" + "="*60)

    # 测试API模型（如果有配置）
    if OPENAI_CONFIG.get("api_key"):
        try:
            llm_api = LLMManager(
                model_name="gpt-4",
                use_api=True
            )

            print("API模型初始化成功")

            prompt = "高血压患者应该注意什么？"
            response = llm_api.generate(prompt, streaming=False)

            print(f"\nAPI测试生成:")
            print(f"问题: {prompt}")
            print(f"回答: {response[:200]}...")

        except Exception as e:
            print(f"API模型测试失败: {e}")
    else:
        print("未配置API密钥，跳过API测试")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM管理器测试")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM, help="模型名称")
    parser.add_argument("--use-api", action="store_true", help="使用API")
    parser.add_argument("--prompt", type=str, default="你好", help="测试提示词")
    parser.add_argument("--stream", action="store_true", help="流式生成")

    args = parser.parse_args()

    llm = LLMManager(
        model_name=args.model,
        use_api=args.use_api
    )

    print(f"使用模型: {args.model} ({'API' if args.use_api else '本地'})")

    # 生成测试
    response = llm.generate(
        prompt=args.prompt,
        streaming=args.stream
    )

    if args.stream:
        print(f"\n流式响应:")
        full_response = ""
        for chunk in response:
            print(chunk, end="", flush=True)
            full_response += chunk
        print()
    else:
        print(f"\n响应: {response}")

    # 模型信息
    info = llm.get_model_info()
    print(f"\n模型信息:")
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")