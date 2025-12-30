"""
评估模块
对微调前后的模型进行多维度评估
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import nltk
from nltk.tokenize import word_tokenize


nltk_data_paths = [
    "/root/autodl-tmp/study/nltk_data",  # 你的项目目录
    "D:\\study\\LLM\\medical_rag\\nltk_data",
]

for path in nltk_data_paths:
    if os.path.exists(path):
        nltk.data.path.append(path)
        print(f"添加 NLTK 数据路径: {path}")


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)


class MedicalRAGEvaluator:
    """医疗RAG系统评估器"""

    def __init__(self, rag_system_base, rag_system_finetuned=None):
        """
        初始化评估器

        Args:
            rag_system_base: 基础RAG系统
            rag_system_finetuned: 微调后的RAG系统（可选）
        """
        self.rag_base = rag_system_base
        self.rag_finetuned = rag_system_finetuned

        # 评估指标存储
        self.metrics = {
            "base_model": {},
            "finetuned_model": {},
            "comparison": {}
        }

    def load_test_dataset(self, test_file: str = "./finetune_data/val.json") -> List[Dict]:
        """加载测试数据集"""
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        logger.info(f"加载测试数据: {len(test_data)} 条")
        return test_data

    def evaluate_retrieval_quality(self, test_questions: List[str],
                                   k: int = 5) -> Dict[str, Any]:
        """评估检索质量"""
        logger.info("评估检索质量...")

        metrics = {
            "base_relevance_scores": [],
            "finetuned_relevance_scores": []
        }

        for question in tqdm(test_questions[:50], desc="评估检索"):
            # 基础模型检索
            base_results = self.rag_base.search_similar_cases(question, k=k)
            base_relevance = self._calculate_relevance(question, base_results)
            metrics["base_relevance_scores"].append(base_relevance)

            # 微调模型检索（如果存在）
            if self.rag_finetuned:
                finetuned_results = self.rag_finetuned.search_similar_cases(question, k=k)
                finetuned_relevance = self._calculate_relevance(question, finetuned_results)
                metrics["finetuned_relevance_scores"].append(finetuned_relevance)

        # 计算统计
        if metrics["base_relevance_scores"]:
            metrics["base_avg_relevance"] = np.mean(metrics["base_relevance_scores"])
            metrics["base_std_relevance"] = np.std(metrics["base_relevance_scores"])

        if metrics["finetuned_relevance_scores"]:
            metrics["finetuned_avg_relevance"] = np.mean(metrics["finetuned_relevance_scores"])
            metrics["finetuned_std_relevance"] = np.std(metrics["finetuned_relevance_scores"])

        return metrics

    def _calculate_relevance(self, question: str, results: List[Dict]) -> float:
        """计算检索结果的相关性"""
        if not results:
            return 0.0

        relevance_scores = []
        for result in results:
            content = result["content"]

            # 简单的关键词匹配评分
            question_keywords = set(word_tokenize(question.lower()))
            content_keywords = set(word_tokenize(content.lower()))

            if question_keywords:
                overlap = len(question_keywords.intersection(content_keywords))
                relevance = overlap / len(question_keywords)
                relevance_scores.append(relevance)

        return np.mean(relevance_scores) if relevance_scores else 0.0

    def evaluate_generation_quality(self, test_data: List[Dict]) -> Dict[str, Any]:
        """评估生成质量"""
        logger.info("评估生成质量...")

        metrics = {
            "base_bleu": [],
            "base_rouge": [],
            "base_similarity": [],
            "finetuned_bleu": [],
            "finetuned_rouge": [],
            "finetuned_similarity": []
        }

        rouge = Rouge()

        for item in tqdm(test_data[:20], desc="评估生成"):
            question = item["instruction"]
            ground_truth = item["output"]

            # 基础模型生成
            base_response = self.rag_base.query(question, use_streaming=False)
            base_bleu = self._calculate_bleu(ground_truth, base_response)
            base_rouge = self._calculate_rouge(rouge, ground_truth, base_response)
            base_similarity = self._calculate_semantic_similarity(ground_truth, base_response)

            metrics["base_bleu"].append(base_bleu)
            metrics["base_rouge"].append(base_rouge)
            metrics["base_similarity"].append(base_similarity)

            # 微调模型生成（如果存在）
            if self.rag_finetuned:
                finetuned_response = self.rag_finetuned.query(question, use_streaming=False)
                finetuned_bleu = self._calculate_bleu(ground_truth, finetuned_response)
                finetuned_rouge = self._calculate_rouge(rouge, ground_truth, finetuned_response)
                finetuned_similarity = self._calculate_semantic_similarity(ground_truth, finetuned_response)

                metrics["finetuned_bleu"].append(finetuned_bleu)
                metrics["finetuned_rouge"].append(finetuned_rouge)
                metrics["finetuned_similarity"].append(finetuned_similarity)

        # 计算平均值
        for key in list(metrics.keys()):
            if metrics[key]:
                metrics[f"{key}_avg"] = np.mean(metrics[key])
                metrics[f"{key}_std"] = np.std(metrics[key])

        return metrics

    def _calculate_bleu(self, reference: str, candidate: str) -> float:
        """计算BLEU分数"""
        reference_tokens = [word_tokenize(reference)]
        candidate_tokens = word_tokenize(candidate)

        # 使用平滑函数处理零匹配的情况
        smoothie = SmoothingFunction().method4

        try:
            score = sentence_bleu(reference_tokens, candidate_tokens,
                                  smoothing_function=smoothie)
            return score
        except:
            return 0.0

    def _calculate_rouge(self, rouge_obj, reference: str, candidate: str) -> float:
        """计算ROUGE-L分数"""
        try:
            scores = rouge_obj.get_scores(candidate, reference)
            return scores[0]['rouge-l']['f']
        except:
            return 0.0

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度（基于词重叠）"""
        words1 = set(word_tokenize(text1.lower()))
        words2 = set(word_tokenize(text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def evaluate_medical_accuracy(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """评估医疗准确性（需要医疗专家标注）"""
        logger.info("评估医疗准确性...")

        # 这里需要医疗专家标注数据
        # 为了演示，我们使用一个简单的模拟评估

        medical_keywords = [
            "建议", "治疗", "药物", "手术", "检查", "诊断",
            "康复", "预防", "饮食", "运动", "休息"
        ]

        metrics = {
            "base_medical_keyword_presence": [],
            "finetuned_medical_keyword_presence": []
        }

        for case in tqdm(test_cases[:30], desc="评估医疗准确性"):
            question = case.get("instruction", "")

            # 基础模型
            base_response = self.rag_base.query(question, use_streaming=False)
            base_score = self._check_medical_keywords(base_response, medical_keywords)
            metrics["base_medical_keyword_presence"].append(base_score)

            # 微调模型
            if self.rag_finetuned:
                finetuned_response = self.rag_finetuned.query(question, use_streaming=False)
                finetuned_score = self._check_medical_keywords(finetuned_response, medical_keywords)
                metrics["finetuned_medical_keyword_presence"].append(finetuned_score)

        # 计算统计
        if metrics["base_medical_keyword_presence"]:
            metrics["base_avg_medical_score"] = np.mean(metrics["base_medical_keyword_presence"])

        if metrics["finetuned_medical_keyword_presence"]:
            metrics["finetuned_avg_medical_score"] = np.mean(metrics["finetuned_medical_keyword_presence"])

        return metrics

    def _check_medical_keywords(self, text: str, keywords: List[str]) -> float:
        """检查医疗关键词出现情况"""
        if not text:
            return 0.0

        text_lower = text.lower()
        found_keywords = [kw for kw in keywords if kw in text_lower]

        return len(found_keywords) / len(keywords)

    def evaluate_response_time(self, test_questions: List[str]) -> Dict[str, Any]:
        """评估响应时间"""
        logger.info("评估响应时间...")

        metrics = {
            "base_response_times": [],
            "finetuned_response_times": []
        }

        for question in tqdm(test_questions[:20], desc="评估响应时间"):
            # 基础模型
            import time
            start_time = time.time()
            _ = self.rag_base.query(question, use_streaming=False)
            base_time = time.time() - start_time
            metrics["base_response_times"].append(base_time)

            # 微调模型
            if self.rag_finetuned:
                start_time = time.time()
                _ = self.rag_finetuned.query(question, use_streaming=False)
                finetuned_time = time.time() - start_time
                metrics["finetuned_response_times"].append(finetuned_time)

        # 计算统计
        if metrics["base_response_times"]:
            metrics["base_avg_time"] = np.mean(metrics["base_response_times"])
            metrics["base_std_time"] = np.std(metrics["base_response_times"])

        if metrics["finetuned_response_times"]:
            metrics["finetuned_avg_time"] = np.mean(metrics["finetuned_response_times"])
            metrics["finetuned_std_time"] = np.std(metrics["finetuned_response_times"])

        return metrics

    def comprehensive_evaluation(self, test_file: str = "./finetune_data/val.json"):
        """综合评估"""
        logger.info("开始综合评估...")

        # 加载测试数据
        test_data = self.load_test_dataset(test_file)
        test_questions = [item["instruction"] for item in test_data]

        # 执行各项评估
        retrieval_metrics = self.evaluate_retrieval_quality(test_questions)
        generation_metrics = self.evaluate_generation_quality(test_data)
        medical_metrics = self.evaluate_medical_accuracy(test_data)
        time_metrics = self.evaluate_response_time(test_questions)

        # 汇总结果
        self.metrics["base_model"].update({
            "retrieval": {
                "avg_relevance": retrieval_metrics.get("base_avg_relevance", 0),
                "std_relevance": retrieval_metrics.get("base_std_relevance", 0)
            },
            "generation": {
                "avg_bleu": generation_metrics.get("base_bleu_avg", 0),
                "avg_rouge": generation_metrics.get("base_rouge_avg", 0),
                "avg_similarity": generation_metrics.get("base_similarity_avg", 0)
            },
            "medical_accuracy": {
                "avg_score": medical_metrics.get("base_avg_medical_score", 0)
            },
            "performance": {
                "avg_response_time": time_metrics.get("base_avg_time", 0),
                "std_response_time": time_metrics.get("base_std_time", 0)
            }
        })

        if self.rag_finetuned:
            self.metrics["finetuned_model"].update({
                "retrieval": {
                    "avg_relevance": retrieval_metrics.get("finetuned_avg_relevance", 0),
                    "std_relevance": retrieval_metrics.get("finetuned_std_relevance", 0)
                },
                "generation": {
                    "avg_bleu": generation_metrics.get("finetuned_bleu_avg", 0),
                    "avg_rouge": generation_metrics.get("finetuned_rouge_avg", 0),
                    "avg_similarity": generation_metrics.get("finetuned_similarity_avg", 0)
                },
                "medical_accuracy": {
                    "avg_score": medical_metrics.get("finetuned_avg_medical_score", 0)
                },
                "performance": {
                    "avg_response_time": time_metrics.get("finetuned_avg_time", 0),
                    "std_response_time": time_metrics.get("finetuned_std_time", 0)
                }
            })

            # 计算改进比例
            self.metrics["comparison"] = self._calculate_improvements()

        # 保存结果
        self.save_results()

        return self.metrics

    def _calculate_improvements(self) -> Dict[str, Any]:
        """计算改进比例"""
        base = self.metrics["base_model"]
        finetuned = self.metrics["finetuned_model"]

        improvements = {}

        # 检索质量改进
        base_ret = base["retrieval"]["avg_relevance"]
        ft_ret = finetuned["retrieval"]["avg_relevance"]
        if base_ret > 0:
            improvements["retrieval_improvement"] = (ft_ret - base_ret) / base_ret

        # 生成质量改进
        for metric in ["avg_bleu", "avg_rouge", "avg_similarity"]:
            base_val = base["generation"][metric]
            ft_val = finetuned["generation"][metric]
            if base_val > 0:
                improvements[f"{metric}_improvement"] = (ft_val - base_val) / base_val

        # 医疗准确性改进
        base_med = base["medical_accuracy"]["avg_score"]
        ft_med = finetuned["medical_accuracy"]["avg_score"]
        if base_med > 0:
            improvements["medical_accuracy_improvement"] = (ft_med - base_med) / base_med

        # 性能改进（响应时间减少）
        base_time = base["performance"]["avg_response_time"]
        ft_time = finetuned["performance"]["avg_response_time"]
        if base_time > 0:
            improvements["speed_improvement"] = (base_time - ft_time) / base_time

        return improvements

    def save_results(self, output_dir: str = "./evaluation_results"):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存详细指标
        metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

        # 生成报告
        self.generate_report(output_dir)

        logger.info(f"评估结果已保存到: {output_dir}")

    def generate_report(self, output_dir: str):
        """生成评估报告"""
        report_file = os.path.join(output_dir, "evaluation_report.md")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 医疗RAG系统评估报告\n\n")

            f.write("## 1. 评估概述\n")
            f.write(f"- 评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 基础模型: {self.rag_base.get_system_info().get('model_mode', 'unknown')}\n")

            if self.rag_finetuned:
                f.write(f"- 微调模型: {self.rag_finetuned.get_system_info().get('model_mode', 'unknown')}\n")

            f.write("\n## 2. 评估结果\n\n")

            # 基础模型结果
            f.write("### 2.1 基础模型性能\n")
            f.write("| 指标类别 | 具体指标 | 数值 |\n")
            f.write("|----------|----------|------|\n")

            base_metrics = self.metrics["base_model"]
            for category, metrics in base_metrics.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, dict):
                        for sub_name, sub_value in value.items():
                            f.write(f"| {category} | {sub_name} | {sub_value:.4f} |\n")
                    else:
                        f.write(f"| {category} | {metric_name} | {value:.4f} |\n")

            # 微调模型结果（如果有）
            if self.rag_finetuned:
                f.write("\n### 2.2 微调模型性能\n")
                f.write("| 指标类别 | 具体指标 | 数值 |\n")
                f.write("|----------|----------|------|\n")

                ft_metrics = self.metrics["finetuned_model"]
                for category, metrics in ft_metrics.items():
                    for metric_name, value in metrics.items():
                        if isinstance(value, dict):
                            for sub_name, sub_value in value.items():
                                f.write(f"| {category} | {sub_name} | {sub_value:.4f} |\n")
                        else:
                            f.write(f"| {category} | {metric_name} | {value:.4f} |\n")

                # 改进对比
                f.write("\n### 2.3 改进对比\n")
                f.write("| 改进指标 | 改进比例 |\n")
                f.write("|----------|----------|\n")

                improvements = self.metrics.get("comparison", {})
                for metric, improvement in improvements.items():
                    percentage = improvement * 100
                    f.write(f"| {metric} | {percentage:.2f}% |\n")

            f.write("\n## 3. 结论与建议\n")

            if self.rag_finetuned:
                f.write("\n微调模型在以下方面有显著改进：\n")
                improvements = self.metrics.get("comparison", {})
                for metric, improvement in improvements.items():
                    if improvement > 0.1:  # 大于10%的改进
                        f.write(f"- {metric}: 改进 {improvement * 100:.1f}%\n")

            f.write("\n**建议**：\n")
            f.write("1. 进一步优化检索策略\n")
            f.write("2. 增加医疗专业知识验证\n")
            f.write("3. 考虑多轮对话评估\n")

        # 生成可视化图表
        self._generate_plots(output_dir)


def run_evaluation(base_rag_system, finetuned_rag_system=None):
    """运行评估的主函数"""
    logger.info("开始评估医疗RAG系统...")

    evaluator = MedicalRAGEvaluator(base_rag_system, finetuned_rag_system)
    metrics = evaluator.comprehensive_evaluation()

    print("\n" + "=" * 60)
    print("📊 评估结果摘要")
    print("=" * 60)

    print("\n🔍 检索质量:")
    print(f"  基础模型相关性: {metrics['base_model']['retrieval']['avg_relevance']:.4f}")
    if finetuned_rag_system:
        print(f"  微调模型相关性: {metrics['finetuned_model']['retrieval']['avg_relevance']:.4f}")

    print("\n💬 生成质量:")
    print(f"  基础模型BLEU: {metrics['base_model']['generation']['avg_bleu']:.4f}")
    print(f"  基础模型ROUGE-L: {metrics['base_model']['generation']['avg_rouge']:.4f}")

    if finetuned_rag_system:
        print(f"  微调模型BLEU: {metrics['finetuned_model']['generation']['avg_bleu']:.4f}")
        print(f"  微调模型ROUGE-L: {metrics['finetuned_model']['generation']['avg_rouge']:.4f}")

    print("\n⚕️ 医疗准确性:")
    print(f"  基础模型医疗评分: {metrics['base_model']['medical_accuracy']['avg_score']:.4f}")
    if finetuned_rag_system:
        print(f"  微调模型医疗评分: {metrics['finetuned_model']['medical_accuracy']['avg_score']:.4f}")

    if finetuned_rag_system and metrics.get('comparison'):
        print("\n📈 改进对比:")
        for metric, improvement in metrics['comparison'].items():
            print(f"  {metric}: {improvement * 100:+.2f}%")

    return metrics


if __name__ == "__main__":
    # 示例使用
    from rag_system import EnhancedMedicalRAGSystem

    # 初始化基础RAG系统
    base_rag = EnhancedMedicalRAGSystem(model_mode="openai")
    base_rag.initialize()

    # 初始化微调RAG系统（如果存在）
    finetuned_rag = None
    finetuned_model_path = "./models/finetuned_medical"

    if os.path.exists(finetuned_model_path):
        finetuned_rag = EnhancedMedicalRAGSystem(
            model_mode="local",
            local_model_path=finetuned_model_path,
            use_finetuned=True
        )
        finetuned_rag.initialize()

    # 运行评估
    metrics = run_evaluation(base_rag, finetuned_rag)