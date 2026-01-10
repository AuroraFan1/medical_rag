"""
完整的医疗RAG系统执行脚本
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from config import DATA_DIR, PROCESSED_DIR, MODELS_DIR
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"medical_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_data_processing_pipeline(args):
    """运行数据处理管道"""
    from data_processing import process_medical_texts, FinetuneDataGenerator

    print("="*60)
    print("📂 数据处理管道")
    print("="*60)

    # 处理文本数据
    cases, documents = process_medical_texts(
        data_dir=args.data_dir,
        max_files=args.max_files,
        max_cases_per_file=args.max_cases,
        test_mode=args.test,
        rebuild=args.rebuild
    )

    # 生成微调数据
    if args.generate_finetune and cases:
        finetune_dir = Path("./processed/finetune_data")
        generator = FinetuneDataGenerator(finetune_dir)
        result = generator.generate_from_cases(cases)

        print("\n✅ 微调数据生成完成")
        print(f"训练样本: {result['train_samples']}")
        print(f"验证样本: {result['val_samples']}")
        print(f"输出目录: {result['output_dir']}")

    return cases, documents

def run_vector_store_pipeline(args, documents):
    """运行向量数据库管道"""
    from vector_store import create_vector_store_from_documents

    print("\n" + "="*60)
    print("🗄️ 向量数据库管道")
    print("="*60)

    # 转换为向量数据库格式
    vector_docs = []
    for doc in documents:
        vector_docs.append({
            "id": doc.metadata.get("id", f"doc_{hash(doc.page_content) % 1000000}"),
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    # 创建向量数据库
    vector_store = create_vector_store_from_documents(
        documents=vector_docs,
        store_name=args.store_name,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        rebuild=args.rebuild
    )

    # 显示信息
    info = vector_store.get_collection_info()
    print(f"\n✅ 向量数据库创建完成")
    print(f"文档数量: {info['document_count']}")
    print(f"嵌入模型: {info['embedding_model']}")
    print(f"存储路径: {info['storage_path']}")

    return vector_store

def run_finetuning_pipeline(args):
    """运行微调管道"""
    from fine_tuner import run_finetuning_pipeline as run_ft_pipeline

    print("\n" + "="*60)
    print("🤖 模型微调管道")
    print("="*60)

    # 检查数据文件
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)

    if not train_path.exists():
        print(f"❌ 训练文件不存在: {train_path}")
        return None

    if not val_path.exists():
        print(f"❌ 验证文件不存在: {val_path}")
        return None

    # 运行微调
    result = run_ft_pipeline(
        train_path=args.train_path,
        val_path=args.val_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        test_mode=args.test,
        merge_model=args.merge
    )

    if result:
        print(f"\n✅ 模型微调完成")
        print(f"输出目录: {result['output_dir']}")
        print(f"训练损失: {result['metrics'].get('train_loss', 'N/A')}")
        print(f"评估损失: {result['metrics'].get('eval_loss', 'N/A')}")

        if "merged_model_path" in result:
            print(f"完整模型: {result['merged_model_path']}")

    return result

def run_rag_system_pipeline(args):
    """运行RAG系统管道"""
    from medical_rag import build_complete_rag_pipeline

    print("\n" + "="*60)
    print("🏥 RAG系统管道")
    print("="*60)

    # 构建完整RAG系统
    rag_system = build_complete_rag_pipeline(
        data_dir=args.data_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        vector_store_name=args.store_name,
        test_mode=args.test,
        rebuild=args.rebuild
    )

    if rag_system:
        print(f"\n✅ RAG系统构建完成")

        # 测试查询
        if args.test_query:
            print(f"\n🔍 测试查询: {args.test_query}")
            result = rag_system.query(args.test_query, use_streaming=False)

            print(f"\n响应: {result['response'][:200]}...")
            print(f"来源数: {result['metadata']['sources_count']}")
            print(f"响应时间: {result['metadata']['response_time']:.2f}s")

        # 系统信息
        system_info = rag_system.get_system_info()
        print(f"\n📊 系统信息:")
        print(f"嵌入模型: {system_info['components']['embedder']['model_name']}")
        print(f"LLM模型: {system_info['components']['llm']['model_name']}")
        print(f"向量数据库: {system_info['components']['vector_store']['document_count']} 文档")

    return rag_system

def run_evaluation_pipeline(args, base_rag, finetuned_rag):
    """运行评估管道"""
    from evaluator import MedicalRAGEvaluator

    print("\n" + "="*60)
    print("📊 评估管道")
    print("="*60)

    # 检查测试数据
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        print(f"❌ 测试数据不存在: {test_data_path}")
        return None

    # 创建评估器
    evaluator = MedicalRAGEvaluator(
        base_rag_system=base_rag,
        finetuned_rag_system=finetuned_rag
    )

    # 运行评估
    results = evaluator.run_comprehensive_evaluation(
        test_dataset_path=args.test_data,
        output_dir=Path(args.eval_output) if args.eval_output else None,
        max_samples=args.max_eval_samples
    )

    if results:
        print(f"\n✅ 评估完成")

        # 打印结果摘要
        for model_name in ["base_model", "finetuned_model"]:
            if model_name in results and results[model_name]:
                metrics = results[model_name]["average_metrics"]
                print(f"\n{model_name}:")
                print(f"  ROUGE-L: {metrics.get('rougeL', 0):.4f}")
                print(f"  幻觉率: {metrics.get('hallucination_rate', 0):.4f}")
                print(f"  引用F1: {metrics.get('citation_f1', 0):.4f}")
                print(f"  医疗准确性: {metrics.get('medical_accuracy', 0):.4f}")

        if results.get("comparison"):
            print("\n📈 模型改进:")
            for metric, improvement in results["comparison"].items():
                if "improvement" in metric:
                    print(f"  {metric}: {improvement*100:+.2f}%")

    return results

def run_full_pipeline(args):
    """运行完整管道"""

    results = {}

    # 1. 数据处理
    if args.run_data_processing:
        #args.data_dir = DATA_DIR
        cases, documents = run_data_processing_pipeline(args)
        results["data_processing"] = {
            "cases": len(cases) if cases else 0,
            "documents": len(documents) if documents else 0
        }

    # 2. 向量数据库
    if args.run_vector_store and "documents" in locals() and documents:
        vector_store = run_vector_store_pipeline(args, documents)
        if vector_store:
            results["vector_store"] = vector_store.get_collection_info()

    # 3. 模型微调
    if args.run_finetuning:
        finetune_result = run_finetuning_pipeline(args)
        if finetune_result:
            results["finetuning"] = finetune_result

    # 4. RAG系统
    if args.run_rag_system:
        rag_system = run_rag_system_pipeline(args)
        if rag_system:
            results["rag_system"] = rag_system.get_system_info()

    # 5. 评估
    if args.run_evaluation:
        from medical_rag import build_complete_rag_pipeline
        # 这里需要实际的RAG系统实例
        # 简化处理：创建两个测试系统
        base_rag = None
        finetuned_rag = None

        if base_rag is None:
            print("正在创建基础RAG系统...")
            base_rag = build_complete_rag_pipeline(
                data_dir=args.data_dir,
                embedding_model=args.embedding_model,
                llm_model=args.base_model,  # 使用基础模型
                vector_store_name=args.store_name,
                test_mode=args.test,
                rebuild=False  # 评估时不需要重建
            )

        if finetuned_rag is None:
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
            model_path = Path(finetuned_model_path)
            print(f"<UNK>: {model_path}")
            finetuned_rag = build_complete_rag_pipeline(
                data_dir=args.data_dir,
                embedding_model=args.embedding_model,
                llm_model=model_path,  # 使用微调模型路径
                vector_store_name=args.store_name,
                test_mode=args.test,
                rebuild=False
            )


        eval_results = run_evaluation_pipeline(args, base_rag, finetuned_rag)
        if eval_results:
            results["evaluation"] = eval_results

    # 保存管道结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"pipeline_results_{timestamp}.json")

    import json
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完整管道执行完成")
    print(f"结果保存到: {results_file}")

    return results

def main():
    """主函数"""

    parser = argparse.ArgumentParser(description="医疗RAG系统完整管道")

    # 阶段选择
    parser.add_argument("--run-all", action="store_true", help="运行所有阶段")
    parser.add_argument("--run-data-processing", action="store_true", help="运行数据处理")
    parser.add_argument("--run-vector-store", action="store_true", help="运行向量数据库")
    parser.add_argument("--run-finetuning", action="store_true", help="运行模型微调")
    parser.add_argument("--run-rag-system", action="store_true", help="运行RAG系统")
    parser.add_argument("--run-evaluation", action="store_true", help="运行评估")

    # 数据处理参数
    parser.add_argument("--data-dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--max-files", type=int, help="最大处理文件数")
    parser.add_argument("--max-cases", type=int, default=100000, help="每个文件最大病例数")
    parser.add_argument("--generate-finetune", action="store_true", help="生成微调数据")

    # 向量数据库参数
    parser.add_argument("--store-name", type=str, default="medical_cases_v1", help="向量数据库名称")
    parser.add_argument("--embedding-model", type=str, default="bge-large-zh", help="嵌入模型")
    parser.add_argument("--batch-size", type=int, default=100, help="批处理大小")

    # 微调参数
    parser.add_argument("--train-path", type=str, default="./processed/finetune_data/train.jsonl", help="训练数据路径")
    parser.add_argument("--val-path", type=str, default="./processed/finetune_data/val.jsonl", help="验证数据路径")
    parser.add_argument("--base-model", type=str, default="qwen2.5-1.5b", help="基础模型")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--merge", action="store_true", help="合并完整模型")

    # RAG系统参数
    parser.add_argument("--llm-model", type=str, default="qwen2.5-1.5b", help="LLM模型")
    parser.add_argument("--test-query", type=str, help="测试查询")

    # 评估参数
    parser.add_argument("--test-data", type=str, default="./data/test_from_pkl.json", help="测试数据路径")
    parser.add_argument("--max-eval-samples", type=int, default=5000, help="最大评估样本数")
    parser.add_argument("--eval-output", type=str, help="评估输出目录")

    # 通用参数
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--rebuild", action="store_true", help="重新构建")

    args = parser.parse_args()

    # 如果指定了--run-all，则运行所有阶段
    if args.run_all:
        args.run_data_processing = True
        args.run_vector_store = True
        args.run_finetuning = True
        args.run_rag_system = True
        args.run_evaluation = True
        args.generate_finetune = True
        args.merge = True

    # 执行管道
    results = run_full_pipeline(args)

    # 打印摘要
    print("\n" + "="*60)
    print("📋 管道执行摘要")
    print("="*60)

    for stage, result in results.items():
        if isinstance(result, dict):
            print(f"{stage}:")
            for key, value in result.items():
                if isinstance(value, (int, float, str, bool)):
                    print(f"  {key}: {value}")
        else:
            print(f"{stage}: {result}")

    print("\n✅ 所有任务完成！")

if __name__ == "__main__":
    main()