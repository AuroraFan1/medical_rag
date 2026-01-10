"""
医疗RAG系统配置文件 - 针对文本文件处理优化
"""

import os
import sys
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data/MedDialog"  # 原始文本文件存放目录
PROCESSED_DIR = BASE_DIR / "processed"  # 处理后的数据
MODELS_DIR = BASE_DIR / "models"  # 模型保存目录
VECTOR_STORE_DIR = BASE_DIR / "vector_store"  # 向量数据库
RESULTS_DIR = BASE_DIR / "results"  # 评估结果
EMBEDDING_MODELS_DIR = MODELS_DIR / "embedding_models"

# 创建目录
for dir_path in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, VECTOR_STORE_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== 硬件配置 ====================
DEVICE = "cuda"
GPU_MEMORY = 32  # RTX 5090 32GB
BATCH_SIZE_EMBEDDING = 32  # 嵌入批量大小
BATCH_SIZE_FINETUNE = 4    # 微调批量大小
MAX_SEQ_LENGTH = 8192     # 支持长上下文

# ==================== 模型配置 ====================
# 基础模型
MODEL_CONFIGS = {
    "qwen2.5-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "max_length": 32768,
        "quantization": "4bit",
        "trainable_params": "1.5B",
        "suitable_for_finetune": True,
        "local_path": str("/root/autodl-tmp/study/medical——rag/models1")  # 本地路径
    },
    "qwen2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "max_length": 32768,
        "quantization": "4bit",
        "trainable_params": "7B",
        "suitable_for_finetune": True,
        "local_path": str("/root/autodl-tmp/study/models1/qwen/Qwen2.5-7B-Instruct")  # 本地路径
    },
    "qwen2.5-14b": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "max_length": 32768,
        "quantization": "4bit",
        "trainable_params": "14B",
        "suitable_for_finetune": False,  # 14B微调需要更多显存
    }
}

# 嵌入模型（针对中文医疗文本优化）
EMBEDDING_MODELS = {
    "bge-large-zh": {
        "name": "BAAI/bge-large-zh-v1.5",
        "dimension": 1024,
        "max_length": 512,
        "language": "zh",
        "medical_optimized": False,
        "local_path": "/root/autodl-tmp/study/medical——rag/models/bge-large-zh-v1.5"
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "dimension": 1024,
        "max_length": 8192,
        "language": "multilingual",
        "medical_optimized": False
    },
    "text2vec-large-chinese": {
        "name": "GanymedeNil/text2vec-large-chinese",
        "dimension": 1024,
        "max_length": 512,
        "language": "zh",
        "medical_optimized": True
    }
}

# 默认选择
DEFAULT_LLM = "qwen2.5-1.5b"
DEFAULT_EMBEDDER = "bge-large-zh"

# ==================== 数据处理配置 ====================
# 文本文件解析配置
TEXT_PARSING_CONFIG = {
    "file_encoding": "utf-8",
    "fallback_encoding": "gbk",
    "max_file_size_mb": 1000,  # 最大文件大小
    "year_patterns": [r"(\d{4})\.txt$", r"(\d{4})_.*\.txt$"],  # 年份匹配模式
    "case_delimiter": r'\n(?=id=|\d+\.\s|病例\s*[：:])',  # 病例分隔符
    "field_patterns": {
        "disease": [r'疾病\s*[：:]\s*(.*?)(?:\n|$)',
                   r'病情\s*[：:]\s*(.*?)(?:\n|$)',
                   r'诊断\s*[：:]\s*(.*?)(?:\n|$)'],
        "symptoms": [r'症状\s*[：:]\s*(.*?)(?:\n医生|\nDoctor|\n$)',
                    r'描述\s*[：:]\s*(.*?)(?:\n医生|\nDoctor|\n$)',
                    r'主诉\s*[：:]\s*(.*?)(?:\n医生|\nDoctor|\n$)'],
        "doctor_reply": [r'医生\s*[：:]\s*(.*?)(?:\nid=|\n\d+\.|\Z)',
                        r'Doctor[^:]*:\s*(.*?)(?:\nid=|\n\d+\.|\Z)',
                        r'建议\s*[：:]\s*(.*?)(?:\nid=|\n\d+\.|\Z)'],
        "hospital": [r'医院\s*[：:]\s*(.*?)(?:\n科室|\n$)',
                    r'Hospital\s*:\s*(.*?)(?:\n$)'],
        "department": [r'科室\s*[：:]\s*(.*?)(?:\n|$)',
                      r'Department\s*:\s*(.*?)(?:\n|$)']
    }
}

# 分块配置
CHUNKING_CONFIG = {
    "strategy": "recursive",  # recursive, sliding_window, sentence
    "chunk_size": 512,
    "chunk_overlap": 128,
    "separators": ["\n\n", "\n", "。", "；", "？", "！", "，", "、", " "],
    "min_chunk_size": 50,
    "max_chunk_size": 1000
}

# ==================== 微调配置 ====================
FINETUNE_CONFIG = {
    # LoRA配置
    "lora": {
        "r": 8,  # LoRA秩
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },

    # 训练参数（针对RTX 5090 32GB优化）
    "training": {
        "num_epochs": 3,
        "per_device_train_batch_size": BATCH_SIZE_FINETUNE,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "warmup_steps": 50,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 0.5,
        "logging_steps": 10,
        "eval_steps": 1000,
        "save_steps": 500,
        "save_total_limit": 3,
        "fp16": True,
        "bf16": False,
        "max_seq_length": 1024
    },

    # 数据配置
    "data": {
        "max_train_samples": 5000,
        "max_eval_samples": 500,
        "train_test_split": 0.9,
        "instruction_template": "基于以下医疗信息回答问题：\n{context}\n\n问题：{question}"
    },

    # 量化配置（针对RTX 5090优化）
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True
    }
}

# ==================== 检索配置 ====================
RETRIEVAL_CONFIG = {
    "top_k": 10,
    "score_threshold": 0.5,
    "enable_reranking": True,
    "reranker_model": "BAAI/bge-reranker-large",
    "enable_diversity": True,
    "diversity_threshold": 0.8,
    "hybrid_search": {
        "enable": True,
        "bm25_weight": 0.3,
        "semantic_weight": 0.7
    },
    "max_context_length": 4000  # 检索上下文最大长度
}

# ==================== 评估配置 ====================
EVALUATION_CONFIG = {
    "metrics": {
        "accuracy": {
            "compute": True,
            "weighted": True
        },
        "f1_score": {
            "compute": True,
            "average": "macro"
        },
        "hallucination_rate": {
            "compute": True,
            "threshold": 0.3
        },
        "citation_f1": {
            "compute": True,
            "strict": False
        },
        "rouge": {
            "compute": True,
            "metrics": ["rouge1", "rouge2", "rougeL"]
        },
        "bertscore": {
            "compute": True,
            "model_type": "bert-base-chinese"
        },
        "response_time": {
            "compute": True,
            "warmup_queries": 3
        }
    },

    "benchmarks": {
        "cmedqa": {
            "path": str(DATA_DIR / "cmedqa"),
            "format": "json",
            "split": "test"
        },
        "internal_test": {
            "path": str(DATA_DIR / "test_questions.json"),
            "format": "json",
            "questions": 100
        }
    }
}

# ==================== Prompt模板 ====================
PROMPT_TEMPLATES = {
    "base": """你是一个专业的医疗助手，基于医疗知识库提供准确信息。

【参考病例信息】：
{context}

【用户问题】：
{question}

【回答要求】：
1. 基于参考病例提供准确信息，明确标注来源 [来源1], [来源2]...
2. 如果参考信息不足或不确定，明确说明"根据现有病例无法确定"
3. 避免编造信息，保持专业严谨
4. 提供必要的医疗建议和注意事项
5. 提醒用户咨询专业医生

请用中文回答：""",

    "multi_turn": """[对话历史]
{history}

【参考病例信息】：
{context}

【当前问题】：
{question}

请结合对话历史和参考病例信息回答：""",

    "uncertain": """根据提供的病例信息，我无法找到足够的信息来准确回答这个问题。

相关病例涉及的内容：
{context_summary}

但无法针对"{question}"提供确切回答。

建议：
1. 咨询专业医疗人员获取准确诊断
2. 提供更详细的症状描述和检查结果
3. 如有紧急情况立即就医

【重要提醒】：本回答基于有限病例信息，不能替代专业医疗诊断。"""
}

# ==================== 系统配置 ====================
SYSTEM_CONFIG = {
    "enable_streaming": True,
    "enable_cache": True,
    "cache_ttl": 3600,
    "max_conversation_turns": 10,
    "max_tokens_per_response": 2000,
    "temperature": 0.1,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.1
}

# ==================== API配置 ====================
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", "sk-pxnrsirqipkyzvtwoptnypmwtoduzkfvtgnjjyelasfcmiyj"),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
    "model": "gpt-4-turbo-preview"
}

# ==================== 日志配置 ====================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(RESULTS_DIR / "system.log"),
    "max_file_size": 10485760,  # 10MB
    "backup_count": 5
}

# ==================== Streamlit应用配置 ====================

# 应用信息
APP_TITLE = "MedAssist - 医疗智能问答系统"
APP_ICON = "🏥"
APP_VERSION = "2.0"

# 示例问题
EXAMPLE_QUESTIONS = [
    "糖尿病患者应该注意什么饮食？",
    "高血压的常见症状有哪些？",
    "感冒和流感的区别是什么？",
    "胃痛应该怎么处理？",
    "头痛伴有恶心可能是什么原因？",
    "如何预防心血管疾病？",
    "儿童发烧应该怎么办？",
    "孕妇饮食有哪些注意事项？",
    "失眠有哪些治疗方法？",
    "运动损伤后应该怎么处理？"
]

# 免责声明
DISCLAIMER_TEXT = """
⚠️ **重要提醒**

本系统提供的信息基于医疗病例数据库，仅供参考和教育目的，不能替代专业医疗建议、诊断或治疗。 

**使用须知：**
1. 咨询内容不能替代执业医师的面对面诊断
2. 紧急情况请立即就医或拨打急救电话120
3. 使用本系统即表示您理解并接受以上条款

**数据来源：** MedDialog数据库，包含120万中文医疗对话
**更新时间：** 2024年1月
"""

# 侧边栏配置
SIDEBAR_CONFIG = {
    "title": "⚙️ 系统控制面板",
    "model_section_title": "🤖 模型配置",
    "retrieval_section_title": "🔍 检索配置",
    "examples_section_title": "💡 常见问题示例",
    "stats_section_title": "📈 对话统计"
}

# 欢迎消息
WELCOME_MESSAGE = """
您好！我是MedAssist医疗智能助手，基于120万真实医疗病例为您提供参考信息。

**我能为您提供：**
- 疾病症状咨询
- 治疗方案参考
- 医疗建议分析
- 病例相似性检索

**使用提示：**
1. 描述具体的症状或问题
2. 提供相关病史信息
3. 系统会检索相似病例并提供参考建议

⚠️ **重要提醒**：以下信息基于历史病例数据，仅供参考，不能替代专业医疗诊断。
"""

# 紧急提醒
EMERGENCY_NOTICE = """
🚨 **紧急情况提醒**

如有以下情况，请立即就医或拨打急救电话 120：
- 剧烈胸痛、呼吸困难
- 严重外伤、大出血
- 意识丧失、抽搐
- 突发性剧烈头痛
- 药物中毒、过敏反应

**急救电话：120**
"""

# 页脚信息
FOOTER_INFO = {
    "version": "MedAssist v2.0",
    "data_source": "MedDialog数据库 (120万医疗病例)",
    "model": "基于Qwen2.5-7B-Instruct",
    "support": "AI医疗实验室",
    "contact": "contact@medical-ai.com",
    "update_date": "2024年1月"
}

# 聊天消息样式
CHAT_STYLES = {
    "user": {
        "background": "#f0f8ff",
        "border_color": "#2196F3",
        "icon": "👤"
    },
    "assistant": {
        "background": "#f9f9f9",
        "border_color": "#4CAF50",
        "icon": "🤖"
    }
}

# 评估指标说明
METRIC_DESCRIPTIONS = {
    "response_time": "系统处理查询并生成回答所需的时间",
    "sources_count": "回答所参考的医疗病例数量",
    "certainty": "系统对回答准确性的置信度",
    "similarity_score": "检索病例与问题的相似度"
}

# 检索策略说明
RETRIEVAL_STRATEGIES = {
    "hybrid": "结合语义检索和关键词检索，提高召回率",
    "rerank": "使用重排序模型对检索结果进行优化排序",
    "diversity": "确保检索结果的多样性，避免重复信息"
}