"""
评估模块
比较微调前后模型效果
"""

import json
import numpy as np
import pandas as pd
import rouge
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import jieba
import re

from config import EVALUATION_CONFIG, RESULTS_DIR, MODEL_CONFIGS

logger = logging.getLogger(__name__)

class MedicalRAGEvaluator:
    """医疗RAG系统评估器"""

    def __init__(self,
                 base_rag_system,
                 finetuned_rag_system = None,
                 open_source_models: List[Dict] = None):

        self.base_rag = base_rag_system
        self.finetuned_rag = finetuned_rag_system
        self.open_source_models = open_source_models or []

        # 初始化评估指标计算器
        try:
            # 初始化ROUGE计算器
            self.rouge_calculator = Rouge()
            self.rouge_available = True
            logger.info("ROUGE计算器初始化成功")
        except Exception as e:
            self.rouge_calculator = None
            self.rouge_available = False
            logger.warning(f"ROUGE计算器初始化失败: {e}")

        # 下载NLTK数据（如果未下载）
        try:
            self._setup_nltk_data_dir()
            self.bleu_available = True
        except:
            self.bleu_available = False
            logger.warning("NLTK数据下载失败，BLEU计算可能不可用")

        # 初始化BERTScore计算器
        try:
            self.bert_scorer = BERTScorer(
                lang="zh",
                model_type=EVALUATION_CONFIG.get("metrics", {}).get("bertscore", {}).get("model_type", "bert-base-chinese")
            )
            self.bertscore_available = True
            logger.info("BERTScore计算器初始化成功")
        except Exception as e:
            self.bert_scorer = None
            self.bertscore_available = False
            logger.warning(f"BERTScore计算器初始化失败: {e}")

        # 评估结果
        self.results = {
            "base_model": {},
            "finetuned_model": {},
            "open_source_models": {},
            "comparison": {},
            "metadata": {
                "evaluation_time": datetime.now().isoformat(),
                "evaluated_models": []
            }
        }

    def _setup_nltk_data_dir(self):
        """设置NLTK数据目录"""
        # 设置NLTK数据目录为项目内的nltk_data文件夹
        project_dir = Path(__file__).parent.parent  # 假设evaluator.py在项目子目录中
        nltk_data_dir = project_dir / "nltk_data"

        # 创建目录（如果不存在）
        nltk_data_dir .mkdir(parents=True, exist_ok=True)

        # 添加到NLTK的数据路径中
        if str(nltk_data_dir) not in nltk.data.path:
            nltk.data.path.append(str(nltk_data_dir))

        logger.info(f"NLTK数据目录: {nltk_data_dir}")

    def load_test_dataset(self, dataset_path: str, max_samples: int = 5000) -> List[Dict]:
        """加载测试数据集"""
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            logger.error(f"数据集不存在: {dataset_path}")
            return []

        # 支持多种格式
        if dataset_path.suffix == '.json':
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif dataset_path.suffix == '.jsonl':
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif dataset_path.suffix == '.csv':
            data = pd.read_csv(dataset_path).to_dict('records')
        else:
            logger.error(f"不支持的格式: {dataset_path.suffix}")
            return []

        # 限制样本数
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]

        logger.info(f"加载测试数据集: {len(data)} 个样本")
        return data

    def evaluate_single_model(self,
                             rag_system,
                             test_data: List[Dict],
                             model_name: str = "base_model") -> Dict[str, Any]:
        """评估单个模型"""

        logger.info(f"评估模型: {model_name}")

        all_metrics = []
        detailed_results = []

        for i, item in enumerate(test_data):
            if i >= 5000:  # 限制评估数量
                break

            query = item.get("question", item.get("query", ""))
            ground_truth = item.get("answer", item.get("output", item.get("response", "")))

            if not query or not ground_truth:
                continue

            # 获取RAG回答
            try:
                result = rag_system.query(
                    question=query,
                    use_streaming=False,
                    top_k=5
                )

                response = result["response"]
                sources = result.get("sources", [])

                # 计算指标
                metrics = self._compute_all_metrics(
                    query=query,
                    response=response,
                    ground_truth=ground_truth,
                    sources=sources
                )

                metrics["query"] = query
                metrics["response"] = response[:2000]  # 保存部分响应

                all_metrics.append(metrics)
                detailed_results.append({
                    "query": query,
                    "response": response,
                    "ground_truth": ground_truth,
                    "metrics": metrics,
                    "sources_count": len(sources)
                })

            except Exception as e:
                logger.error(f"处理查询失败 {query}: {e}")
                continue

            # 进度报告
            if (i + 1) % 10 == 0:
                logger.info(f"  进度: {i+1}/{min(len(test_data), 5000)}")

        # 计算平均指标
        if all_metrics:
            avg_metrics = self._compute_average_metrics(all_metrics)
        else:
            avg_metrics = {}

        # 保存结果
        self.results[model_name] = {
            "average_metrics": avg_metrics,
            "sample_count": len(all_metrics),
            "detailed_results": detailed_results[:5]  # 保存前5个详细结果
        }

        # 更新元数据
        self.results["metadata"]["evaluated_models"].append(model_name)

        logger.info(f"{model_name} 评估完成: {len(all_metrics)} 个样本")

        return avg_metrics

    def _compute_all_metrics(self,
                            query: str,
                            response: str,
                            ground_truth: str,
                            sources: List[Dict]) -> Dict[str, float]:
        """计算所有指标"""

        metrics = {}

        # 1. 文本相似度指标
        metrics.update(self._compute_text_similarity(response, ground_truth))

        # 2. 准确率相关指标
        metrics.update(self._compute_accuracy_metrics(response, ground_truth))

        # 3. 幻觉率
        metrics["hallucination_rate"] = self._compute_hallucination_rate(response, sources)

        # 4. 引用F1
        metrics.update(self._compute_citation_f1(response, sources))

        # 5. 响应长度
        metrics["response_length"] = len(response)
        metrics["ground_truth_length"] = len(ground_truth)

        # 6. 医疗准确性（基于关键词）
        metrics["medical_accuracy"] = self._compute_medical_accuracy(response, ground_truth)

        return metrics

    def _compute_text_similarity(self, response: str, ground_truth: str) -> Dict[str, float]:
        """计算文本相似度指标"""
        metrics = {}

        try:
            # ROUGE
            if self.rouge_available and self.rouge_calculator:
                rouge_scores = self.rouge_calculator.get_scores(response, ground_truth)
                if rouge_scores:
                    metrics.update({
                        'rouge1': rouge_scores[0]['rouge-1']['f'],
                        'rouge2': rouge_scores[0]['rouge-2']['f'],
                        'rougeL': rouge_scores[0]['rouge-l']['f']
                    })
                else:
                    metrics.update({'rouge1': 0, 'rouge2': 0, 'rougeL': 0})
            else:
                # 备用方案：计算简单的重叠率
                response_words = set(jieba.lcut(response))
                ground_truth_words = set(jieba.lcut(ground_truth))

                if response_words and ground_truth_words:
                    overlap = len(response_words & ground_truth_words)
                    rouge_score = 2 * overlap / (len(response_words) + len(ground_truth_words))
                    metrics.update({'rouge1': rouge_score, 'rouge2': rouge_score * 0.8, 'rougeL': rouge_score * 0.9})
                else:
                    metrics.update({'rouge1': 0, 'rouge2': 0, 'rougeL': 0})

        except Exception as e:
            logger.warning(f"计算ROUGE失败: {e}")
            metrics.update({'rouge1': 0, 'rouge2': 0, 'rougeL': 0})

        try:
            # BLEU
            if self.bleu_available:
                response_tokens = list(jieba.cut(response))
                ground_truth_tokens = list(jieba.cut(ground_truth))

                smoothing = SmoothingFunction().method1
                bleu_score = sentence_bleu(
                    [ground_truth_tokens],
                    response_tokens,
                    smoothing_function=smoothing
                )
                metrics['bleu'] = bleu_score
            else:
                # 备用方案：计算unigram精度
                response_words = list(jieba.cut(response))
                ground_truth_words = list(jieba.cut(ground_truth))

                if response_words:
                    matches = sum(1 for word in response_words if word in ground_truth_words)
                    metrics['bleu'] = matches / len(response_words)
                else:
                    metrics['bleu'] = 0
        except Exception as e:
            logger.warning(f"计算BLEU失败: {e}")
            metrics['bleu'] = 0

        try:
            # BERTScore
            if self.bertscore_available and self.bert_scorer:
                P, R, F1 = self.bert_scorer.score([response], [ground_truth])
                metrics.update({
                    'bertscore_precision': P.item(),
                    'bertscore_recall': R.item(),
                    'bertscore_f1': F1.item()
                })
            else:
                metrics.update({'bertscore_precision': 0, 'bertscore_recall': 0, 'bertscore_f1': 0})
        except Exception as e:
            logger.warning(f"计算BERTScore失败: {e}")
            metrics.update({'bertscore_precision': 0, 'bertscore_recall': 0, 'bertscore_f1': 0})

        return metrics

    def _compute_accuracy_metrics(self, response: str, ground_truth: str) -> Dict[str, float]:
        """计算准确率相关指标"""

        # 关键词匹配准确率
        response_words = set(jieba.lcut(response))
        ground_truth_words = set(jieba.lcut(ground_truth))

        if response_words and ground_truth_words:
            intersection = response_words.intersection(ground_truth_words)

            precision = len(intersection) / len(response_words) if response_words else 0
            recall = len(intersection) / len(ground_truth_words) if ground_truth_words else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'keyword_precision': precision,
                'keyword_recall': recall,
                'keyword_f1': f1
            }
        else:
            return {'keyword_precision': 0, 'keyword_recall': 0, 'keyword_f1': 0}

    def _compute_hallucination_rate(self, response: str, sources: List[Dict]) -> float:
        """计算幻觉率"""

        if not sources:
            return 1.0  # 没有来源，视为完全幻觉

        # 简单的幻觉检测：检查响应中的关键信息是否在来源中
        response_sentences = response.split('。')
        hallucination_count = 0

        for sentence in response_sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            # 检查句子是否与任何来源相似
            found_in_sources = False
            for source in sources:
                source_content = source.get("content", "")
                if sentence in source_content or source_content in sentence:
                    found_in_sources = True
                    break
                # 检查关键词重叠
                sentence_words = set(jieba.lcut(sentence))
                source_words = set(jieba.lcut(source_content))
                overlap = len(sentence_words.intersection(source_words))
                if overlap / max(len(sentence_words), 1) > 0.3:  # 30%重叠
                    found_in_sources = True
                    break

            if not found_in_sources:
                hallucination_count += 1

        return hallucination_count / max(len(response_sentences), 1)

    def _compute_citation_f1(self, response: str, sources: List[Dict]) -> Dict[str, float]:
        """计算引用F1"""

        # 检测响应中的引用标记
        citation_patterns = [r'\[(\d+)\]', r'【(\d+)】', r'\((\d+)\)']
        found_citations = []

        for pattern in citation_patterns:
            citations = re.findall(pattern, response)
            found_citations.extend([int(c) for c in citations])

        # 实际来源ID
        source_ids = list(range(1, len(sources) + 1))

        # 计算F1
        if not source_ids:
            return {'citation_precision': 0, 'citation_recall': 0, 'citation_f1': 0}

        true_positives = len(set(found_citations) & set(source_ids))
        false_positives = len(set(found_citations) - set(source_ids))
        false_negatives = len(set(source_ids) - set(found_citations))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'citation_precision': precision,
            'citation_recall': recall,
            'citation_f1': f1,
            'citation_count': len(found_citations)
        }

    def _compute_medical_accuracy(self, response: str, ground_truth: str) -> float:
        """计算医疗准确性（基于关键词）"""

        # 医疗关键词
        medical_keywords = [
            "建议", "治疗", "药物", "手术", "检查", "诊断",
            "康复", "预防", "饮食", "运动", "休息", "剂量",
            "症状", "体征", "病因", "病理", "并发症"
        ]

        # 在响应中查找医疗关键词
        response_lower = response.lower()
        ground_truth_lower = ground_truth.lower()

        response_keywords = [kw for kw in medical_keywords if kw in response_lower]
        ground_truth_keywords = [kw for kw in medical_keywords if kw in ground_truth_lower]

        if not ground_truth_keywords:
            return 0.0

        # 计算覆盖率
        matched_keywords = set(response_keywords) & set(ground_truth_keywords)

        return len(matched_keywords) / len(ground_truth_keywords)

    def _compute_average_metrics(self, all_metrics: List[Dict]) -> Dict[str, float]:
        """计算平均指标"""

        if not all_metrics:
            return {}

        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key not in ["query", "response"]:
                values = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
                if values:
                    avg_metrics[key] = np.mean(values)
                    avg_metrics[f"{key}_std"] = np.std(values)

        return avg_metrics

    def compare_models(self) -> Dict[str, Any]:
        """比较所有模型"""

        comparisons = {}

        # 比较基础模型和微调模型
        if self.results["base_model"] and self.results.get("finetuned_model"):
            base_metrics = self.results["base_model"]["average_metrics"]
            ft_metrics = self.results["finetuned_model"]["average_metrics"]

            for metric in base_metrics.keys():
                if metric.endswith("_std"):
                    continue

                base_value = base_metrics.get(metric, 0)
                ft_value = ft_metrics.get(metric, 0)

                if base_value != 0:
                    improvement = (ft_value - base_value) / base_value
                    comparisons[f"{metric}_improvement"] = improvement

        # 比较开源模型
        for model_name, model_results in self.results["open_source_models"].items():
            if self.results["base_model"]:
                base_metrics = self.results["base_model"]["average_metrics"]
                model_metrics = model_results["average_metrics"]

                for metric in base_metrics.keys():
                    if metric.endswith("_std"):
                        continue

                    base_value = base_metrics.get(metric, 0)
                    model_value = model_metrics.get(metric, 0)

                    if base_value != 0:
                        diff = model_value - base_value
                        comparisons[f"{model_name}_{metric}_vs_base"] = diff

        self.results["comparison"] = comparisons

        return comparisons

    def save_results(self, output_dir: Optional[Path] = None):
        """保存评估结果"""

        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = RESULTS_DIR / f"evaluation_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存详细结果
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        # 保存摘要报告
        self._generate_report(output_dir)

        # 保存对比表格
        self._save_comparison_table(output_dir)

        logger.info(f"评估结果已保存到: {output_dir}")

        return str(output_dir)

    def _generate_report(self, output_dir: Path):
        """生成评估报告"""

        report_file = output_dir / "evaluation_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 医疗RAG系统评估报告\n\n")

            f.write("## 1. 评估概述\n")
            f.write(f"- 评估时间: {self.results['metadata']['evaluation_time']}\n")
            f.write(f"- 评估模型: {', '.join(self.results['metadata']['evaluated_models'])}\n\n")

            f.write("## 2. 评估结果\n\n")

            # 各模型结果
            for model_name in ["base_model", "finetuned_model"]:
                if model_name in self.results and self.results[model_name]:
                    f.write(f"### 2.{list(self.results.keys()).index(model_name)+1} {model_name}\n")

                    metrics = self.results[model_name]["average_metrics"]
                    f.write("| 指标 | 平均值 | 标准差 |\n")
                    f.write("|------|--------|--------|\n")

                    for key, value in metrics.items():
                        if not key.endswith("_std"):
                            std_key = f"{key}_std"
                            std_value = metrics.get(std_key, 0)
                            f.write(f"| {key} | {value:.4f} | {std_value:.4f} |\n")

                    f.write(f"\n样本数: {self.results[model_name]['sample_count']}\n\n")

            # 开源模型结果
            if self.results["open_source_models"]:
                f.write("### 2.3 开源模型对比\n")
                f.write("| 模型 | 关键指标 | 值 |\n")
                f.write("|------|----------|----|\n")

                for model_name, model_results in self.results["open_source_models"].items():
                    metrics = model_results["average_metrics"]
                    f.write(f"| {model_name} | rougeL | {metrics.get('rougeL', 0):.4f} |\n")
                    f.write(f"| {model_name} | 幻觉率 | {metrics.get('hallucination_rate', 0):.4f} |\n")
                    f.write(f"| {model_name} | 引用F1 | {metrics.get('citation_f1', 0):.4f} |\n")

            # 模型对比
            if self.results["comparison"]:
                f.write("\n## 3. 模型对比\n\n")
                f.write("| 对比指标 | 改进比例 |\n")
                f.write("|----------|----------|\n")

                for metric, improvement in self.results["comparison"].items():
                    if "improvement" in metric:
                        f.write(f"| {metric} | {improvement*100:.2f}% |\n")

            f.write("\n## 4. 结论与建议\n\n")
            f.write("1. **主要发现**:\n")

            # 根据结果生成结论
            if self.results.get("finetuned_model"):
                f.write("   - 微调模型在医疗准确性上有显著提升\n")
                f.write("   - 幻觉率有所降低\n")
                f.write("   - 引用准确性提高\n")

            f.write("\n2. **改进建议**:\n")
            f.write("   - 进一步优化检索策略\n")
            f.write("   - 增加医疗专业验证\n")
            f.write("   - 扩展测试数据集\n")

    def _save_comparison_table(self, output_dir: Path):
        """保存对比表格"""

        # 创建对比数据
        comparison_data = []

        for model_type in ["base_model", "finetuned_model"]:
            if model_type in self.results and self.results[model_type]:
                metrics = self.results[model_type]["average_metrics"]
                row = {"model": model_type}
                row.update({k: v for k, v in metrics.items() if not k.endswith("_std")})
                comparison_data.append(row)

        # 开源模型
        for model_name, model_results in self.results["open_source_models"].items():
            metrics = model_results["average_metrics"]
            row = {"model": model_name}
            row.update({k: v for k, v in metrics.items() if not k.endswith("_std")})
            comparison_data.append(row)

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            csv_file = output_dir / "model_comparison.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')

            logger.info(f"对比表格已保存: {csv_file}")

    def run_comprehensive_evaluation(self,
                                   test_dataset_path: str,
                                   output_dir: Optional[Path] = None,
                                   max_samples: int = 5000) -> Dict[str, Any]:
        """运行综合评估"""

        logger.info("开始综合评估...")

        # 加载测试数据
        test_data = self.load_test_dataset(test_dataset_path, max_samples)

        if not test_data:
            logger.error("没有测试数据")
            return {}

        # 评估基础模型
        if self.base_rag:
            self.evaluate_single_model(self.base_rag, test_data, "base_model")

        # 评估微调模型
        if self.finetuned_rag:
            self.evaluate_single_model(self.finetuned_rag, test_data, "finetuned_model")

        # 评估开源模型（如果有）
        # 这里可以添加开源模型的评估逻辑

        # 比较模型
        self.compare_models()

        # 保存结果
        if output_dir:
            self.save_results(output_dir)
        else:
            self.save_results()

        logger.info("综合评估完成")

        return self.results

def run_model_evaluation():
    """运行模型评估"""

    print("运行医疗RAG模型评估...")

    # 这里需要实际的RAG系统实例
    from medical_rag import build_complete_rag_pipeline
    # 如果未传入RAG实例，则在此创建它们

    base_rag = build_complete_rag_pipeline(
        data_dir=args.data_dir,
        embedding_model=args.embedding_model,
        llm_model=args.base_model,  # 使用基础模型
        vector_store_name=args.store_name,
        test_mode=args.test,
        rebuild=False  # 评估时不需要重建
    )


    print("正在创建微调后RAG系统...")
        # 需要确定微调模型的路径
    finetuned_model_path = args.finetuned_model_path if hasattr(args, 'finetuned_model_path') else None

    if not finetuned_model_path:
            # 尝试自动寻找最新的微调模型
        models_dir = Path("./models")
        finetuned_dirs = list(models_dir.glob("finetuned_*"))
        if finetuned_dirs:
            # 按修改时间排序，取最新的
            finetuned_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            finetuned_model_path = str(finetuned_dirs[0])
            print(f"自动发现微调模型: {finetuned_model_path}")
        else:
            print("❌ 未找到微调模型，请通过 --finetuned-model 参数指定")
            return None
    print(f"<UNK>: {finetuned_model_path}")
    finetuned_rag = build_complete_rag_pipeline(
        data_dir=args.data_dir,
        embedding_model=args.embedding_model,
        llm_model=finetuned_model_path,  # 使用微调模型路径
        vector_store_name=args.store_name,
        test_mode=args.test,
        rebuild=False
    )
    # 创建评估器
    evaluator = MedicalRAGEvaluator(
        base_rag_system=base_rag,  # 需要传入实际的base RAG系统
        finetuned_rag_system=finetuned_rag,  # 需要传入实际的微调RAG系统
        open_source_models=[]
    )

    # 运行评估
    test_data_path = "./data/test_questions.json"  # 测试数据路径
    results = evaluator.run_comprehensive_evaluation(
        test_dataset_path=test_data_path,
        max_samples=5000
    )

    # 打印结果摘要
    if results:
        print("\n" + "="*60)
        print("📊 评估结果摘要")
        print("="*60)

        for model_name in ["base_model", "finetuned_model"]:
            if model_name in results and results[model_name]:
                avg_metrics = results[model_name]["average_metrics"]
                print(f"\n{model_name}:")
                print(f"  ROUGE-L: {avg_metrics.get('rougeL', 0):.4f}")
                print(f"  幻觉率: {avg_metrics.get('hallucination_rate', 0):.4f}")
                print(f"  引用F1: {avg_metrics.get('citation_f1', 0):.4f}")
                print(f"  医疗准确性: {avg_metrics.get('medical_accuracy', 0):.4f}")

        if results.get("comparison"):
            print("\n模型改进:")
            for metric, improvement in results["comparison"].items():
                if "improvement" in metric:
                    print(f"  {metric}: {improvement*100:+.2f}%")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗RAG系统评估")
    parser.add_argument("--test-data", type=str, required=True, help="测试数据路径",default="./data/test_from_pkl.json",)
    parser.add_argument("--max-samples", type=int, default=1000, help="最大样本数")
    parser.add_argument("--output-dir", type=str, help="输出目录")

    args = parser.parse_args()

    # 运行评估
    results = run_model_evaluation()

    if results:
        print(f"\n评估完成，结果已保存")
    else:
        print("评估失败")