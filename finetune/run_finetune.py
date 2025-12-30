"""
微调工作流主脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 然后继续原有的导入
from data_processing import MedicalCase
# import __main__
# from data_processing import MedicalCase
# __main__.MedicalCase = MedicalCase

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """微调工作流主函数"""
    parser = argparse.ArgumentParser(description="医疗RAG系统微调工作流")
    parser.add_argument("--prepare-data", action="store_true", help="准备微调数据")
    parser.add_argument("--finetune", action="store_true", help="微调模型")
    parser.add_argument("--evaluate", action="store_true", help="评估模型")
    parser.add_argument("--all", action="store_true", help="执行完整工作流", default=True)
    parser.add_argument("--test", action="store_true", help="测试模式")

    args = parser.parse_args()

    if args.all or args.prepare_data:
        logger.info("步骤1: 准备微调数据")
        from finetune_data import prepare_finetune_data

        config = {
            "max_samples": 1000 if args.test else 10000,
            "test_mode": args.test
        }

        result = prepare_finetune_data(**config)
        logger.info(f"数据准备完成: {result}")

    if args.all or args.finetune:
        logger.info("步骤2: 微调模型")
        from finetune_model import run_finetune

        config = {
            "test": args.test
        }

        finetuned_model_path = "./models/finetuned_medical"

        if not os.path.exists(finetuned_model_path):
            logger.warning("微调模型不存在，跳过评估")
            result = run_finetune(config)
            if result:
                logger.info(f"模型微调完成: {result['output_dir']}")
        else:
            logger.info(f"模型存在于: {finetuned_model_path}")

    if args.all or args.evaluate:
        logger.info("步骤3: 评估模型")

        # 检查模型是否已存在
        base_model_path = None  # 基础模型
        finetuned_model_path = "./models/finetuned_medical"

        if not os.path.exists(finetuned_model_path):
            logger.warning("微调模型不存在，跳过评估")
            return

        from evaluate import run_evaluation
        from rag_system import EnhancedMedicalRAGSystem

        # 初始化基础RAG
        logger.info("初始化基础RAG系统...")
        base_rag = EnhancedMedicalRAGSystem(model_mode="openai")
        base_success = base_rag.initialize()

        if not base_success:
            logger.error("基础RAG系统初始化失败")
            return

        # 初始化微调RAG
        logger.info("初始化微调RAG系统...")
        finetuned_rag = EnhancedMedicalRAGSystem(
            model_mode="local",
            local_model_path=finetuned_model_path,
            use_finetuned=True
        )
        ft_success = finetuned_rag.initialize()

        if not ft_success:
            logger.error("微调RAG系统初始化失败")
            finetuned_rag = None

        # 运行评估
        metrics = run_evaluation(base_rag, finetuned_rag)
        logger.info(f"评估完成，结果保存到 ./evaluation_results/")

    if not any([args.prepare_data, args.finetune, args.evaluate, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()