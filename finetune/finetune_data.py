"""
微调数据准备模块
从医疗数据中提取用于微调的训练对
"""

import os
import json
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

import sys
from pathlib import Path

# 获取当前文件的父目录的父目录（即根目录）
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from config import DATA_DIR
from data_processing import MedicalDataProcessor

logger = logging.getLogger(__name__)


@dataclass
class FineTuneExample:
    """微调数据样本"""
    instruction: str
    input_text: str
    output_text: str
    source_case: Dict[str, Any]


class FineTuneDataPreparer:
    """微调数据准备器"""

    def __init__(self, processor: MedicalDataProcessor):
        self.processor = processor

    def create_training_pairs(self, cases: List) -> List[FineTuneExample]:
        """从病例数据创建训练对"""
        examples = []

        for case in cases:
            # 使用疾病作为instruction
            instruction = f"基于以下医疗信息回答问题："

            # 输入：症状描述 + 上下文信息
            input_text = f"疾病：{case.disease}\n症状：{case.symptoms}\n医院科室：{case.hospital} {case.department}"

            # 输出：医生建议
            output_text = case.doctor_reply

            if (len(output_text.strip()) > 50 and  # 确保有足够内容
                    len(case.symptoms.strip()) > 20 and
                    len(case.disease.strip()) > 2):
                examples.append(FineTuneExample(
                    instruction=instruction,
                    input_text=input_text,
                    output_text=output_text,
                    source_case={
                        'id': case.id,
                        'disease': case.disease,
                        'year': case.year
                    }
                ))

        return examples

    def generate_qa_pairs(self, cases: List, num_samples: int = 5000) -> List[Dict]:
        """生成问答对用于指令微调"""
        qa_pairs = []

        # 常见问题模板
        question_templates = [
            "关于{疾病}，医生通常会给出什么建议？",
            "如果出现{症状}，可能是什么问题？应该怎么办？",
            "对于{疾病}患者，有哪些需要注意的事项？",
            "如何治疗{疾病}？",
            "{疾病}的常见症状有哪些？如何缓解？",
            "医生对{疾病}患者有哪些建议？"
        ]

        for case in cases:
            if not case.disease or not case.doctor_reply:
                continue

            # 从病例生成多个问题
            disease = case.disease.split('，')[0].split('？')[0]  # 清理疾病名称

            for template in random.sample(question_templates, min(2, len(question_templates))):
                try:
                    question = template.format(疾病=disease)

                    # 生成答案（结合医生建议）
                    context = f"根据病例记录，患者症状：{case.symptoms[:200]}..."
                    answer = f"{context}\n\n医生建议：{case.doctor_reply}"

                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "disease": case.disease,
                        "case_id": case.id
                    })

                    if len(qa_pairs) >= num_samples:
                        break

                except:
                    continue

            if len(qa_pairs) >= num_samples:
                break

        return qa_pairs

    def save_training_data(self, examples: List[FineTuneExample],
                           qa_pairs: List[Dict],
                           output_dir: str = "./finetune_data"):
        """保存训练数据"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存微调样本
        examples_data = []
        for ex in examples:
            examples_data.append({
                "instruction": ex.instruction,
                "input": ex.input_text,
                "output": ex.output_text,
                "source": ex.source_case
            })

        with open(os.path.join(output_dir, "fine_tune_examples.json"), 'w', encoding='utf-8') as f:
            json.dump(examples_data, f, ensure_ascii=False, indent=2)

        # 保存QA对
        with open(os.path.join(output_dir, "qa_pairs.json"), 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

        # 转换为Alpaca格式
        alpaca_data = []
        for qa in qa_pairs:
            alpaca_data.append({
                "instruction": qa["question"],
                "input": "",
                "output": qa["answer"]
            })

        with open(os.path.join(output_dir, "alpaca_format.json"), 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

        logger.info(f"训练数据已保存到 {output_dir}")
        logger.info(f"微调样本数: {len(examples_data)}")
        logger.info(f"QA对数: {len(qa_pairs)}")
        logger.info(f"Alpaca格式数据: {len(alpaca_data)}")

        return output_dir

    def prepare_train_val_split(self, data_path: str,
                                val_ratio: float = 0.1,
                                seed: int = 42):
        """准备训练/验证集分割"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 分割数据
        train_data, val_data = train_test_split(
            data, test_size=val_ratio, random_state=seed
        )

        output_dir = os.path.dirname(data_path)

        with open(os.path.join(output_dir, "train.json"), 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, "val.json"), 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        logger.info(f"训练集: {len(train_data)} 条")
        logger.info(f"验证集: {len(val_data)} 条")

        return len(train_data), len(val_data)


def prepare_finetune_data(years: List[int] = None,
                          max_samples: int = 10000,
                          test_mode: bool = False):
    """准备微调数据的主函数"""
    logger.info("开始准备微调数据...")

    # 加载数据
    processor = MedicalDataProcessor()

    if years is None:
        years = list(range(2010, 2021))

    all_cases = []
    all_cases = processor.load_processed_cases(years)
    # for year in years:
    #     cases = processor.load_processed_cases([year])
    #     all_cases.extend(cases)
    #
    #     if test_mode and len(all_cases) > 1000:
    #         break

    logger.info(f"总共加载 {len(all_cases)} 个病例")

    # 准备数据
    preparer = FineTuneDataPreparer(processor)

    # 创建微调样本
    examples = preparer.create_training_pairs(all_cases[:max_samples])

    # 生成QA对
    qa_pairs = preparer.generate_qa_pairs(all_cases[:max_samples],
                                          num_samples=min(max_samples, len(all_cases)))

    # 保存数据
    output_dir = preparer.save_training_data(examples, qa_pairs)

    # 准备训练/验证分割
    alpaca_path = os.path.join(output_dir, "alpaca_format.json")
    train_size, val_size = preparer.prepare_train_val_split(alpaca_path)

    return {
        "output_dir": output_dir,
        "total_cases": len(all_cases),
        "fine_tune_examples": len(examples),
        "qa_pairs": len(qa_pairs),
        "train_size": train_size,
        "val_size": val_size
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="准备微调数据")
    parser.add_argument("--years", type=str, help="年份，用逗号分隔")
    parser.add_argument("--max-samples", type=int, default=10000, help="最大样本数")
    parser.add_argument("--test", action="store_true", help="测试模式")

    args = parser.parse_args()

    years = None
    if args.years:
        years = [int(y) for y in args.years.split(",")]

    result = prepare_finetune_data(
        years=years,
        max_samples=args.max_samples,
        test_mode=args.test
    )

    print("\n" + "=" * 50)
    print("✅ 微调数据准备完成")
    print("=" * 50)
    for key, value in result.items():
        print(f"  {key}: {value}")