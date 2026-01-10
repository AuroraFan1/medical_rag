"""
专门处理医疗文本文件的数据处理模块
基于您提供的原始代码优化
"""

import os
import re
import json
import pickle
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# LangChain相关
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 导入配置
from config import (
    DATA_DIR, PROCESSED_DIR, TEXT_PARSING_CONFIG,
    CHUNKING_CONFIG, LOGGING_CONFIG
)

# 设置日志
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MedicalCase:
    """医疗病例数据结构 - 基于文本文件格式"""
    id: str
    year: int
    disease: str = ""
    symptoms: str = ""
    doctor_reply: str = ""
    hospital: str = ""
    department: str = ""
    url: str = ""
    raw_text: str = ""
    full_text: str = ""

    def __post_init__(self):
        """初始化后生成完整文本"""
        if not self.full_text and self.raw_text:
            self.full_text = self._create_full_text()

    def _create_full_text(self) -> str:
        """创建完整文本表示"""
        parts = []

        if self.disease:
            parts.append(f"疾病：{self.disease}")

        if self.symptoms:
            parts.append(f"症状：{self.symptoms}")

        if self.doctor_reply:
            parts.append(f"医生建议：{self.doctor_reply}")

        if self.hospital:
            parts.append(f"医院：{self.hospital}")

        if self.department:
            parts.append(f"科室：{self.department}")

        # 如果没有提取到字段，使用原始文本
        if not parts and self.raw_text:
            return self.raw_text[:500]  # 限制长度

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_document(self) -> Document:
        """转换为LangChain文档"""
        metadata = {
            "id": self.id,
            "year": self.year,
            "disease": self.disease,
            "hospital": self.hospital,
            "department": self.department,
            "url": self.url,
            "source": f"medical_case_{self.id}",
            "has_symptoms": bool(self.symptoms),
            "has_advice": bool(self.doctor_reply),
            "text_length": len(self.full_text)
        }

        return Document(
            page_content=self.full_text,
            metadata=metadata
        )

class TextFileProcessor:
    """医疗文本文件处理器"""

    def __init__(self, data_dir: str = str(DATA_DIR)):
        self.data_dir = Path(data_dir)
        self.processed_dir = PROCESSED_DIR
        self.processed_dir.mkdir(exist_ok=True)

        # 统计数据
        self.stats = {
            "total_files": 0,
            "total_cases": 0,
            "failed_cases": 0,
            "processed_years": set(),
            "start_time": datetime.now()
        }

        # 编译正则表达式
        self.field_patterns = TEXT_PARSING_CONFIG["field_patterns"]
        self.compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """编译正则表达式模式"""
        for field_name, patterns in self.field_patterns.items():
            compiled = []
            for pattern in patterns:
                try:
                    compiled.append(re.compile(pattern, re.IGNORECASE | re.DOTALL))
                except re.error as e:
                    logger.warning(f"编译正则表达式失败 {pattern}: {e}")
            self.compiled_patterns[field_name] = compiled

    def extract_year_from_filename(self, filename: str) -> Optional[int]:
        """从文件名中提取年份"""
        for pattern in TEXT_PARSING_CONFIG["year_patterns"]:
            match = re.search(pattern, filename)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None

    def read_text_file(self, file_path: Path) -> Optional[str]:
        """读取文本文件，尝试多种编码"""
        encodings = [
            TEXT_PARSING_CONFIG["file_encoding"],
            TEXT_PARSING_CONFIG["fallback_encoding"],
            'utf-8-sig',
            'gb18030',
            'big5'
        ]

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                logger.info(f"成功读取文件 {file_path.name}，编码: {encoding}")
                return content
            except UnicodeDecodeError:
                continue

        logger.error(f"无法读取文件 {file_path}，所有编码都失败")
        return None

    def split_into_cases(self, content: str) -> List[str]:
        """将文本内容分割为单个病例"""
        # 使用配置的分隔符
        delimiter = TEXT_PARSING_CONFIG["case_delimiter"]
        cases = re.split(delimiter, content.strip())

        # 过滤空病例
        cases = [case.strip() for case in cases if case.strip()]

        logger.info(f"分割得到 {len(cases)} 个病例块")
        return cases

    def extract_field(self, text: str, field_name: str) -> str:
        """提取特定字段"""
        if field_name not in self.compiled_patterns:
            return ""

        patterns = self.compiled_patterns[field_name]
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                # 提取匹配的文本并清理
                extracted = match.group(1).strip()
                # 移除多余的空白字符
                extracted = re.sub(r'\s+', ' ', extracted)
                return extracted[:500]  # 限制长度

        return ""

    def parse_single_case(self, case_text: str, year: int, case_index: int) -> Optional[MedicalCase]:
        """解析单个病例文本"""
        try:
            # 生成病例ID
            case_hash = hashlib.md5(case_text.encode()).hexdigest()[:12]
            case_id = f"{year}_{case_index}_{case_hash}"

            # 提取各个字段
            disease = self.extract_field(case_text, "disease")
            symptoms = self.extract_field(case_text, "symptoms")
            doctor_reply = self.extract_field(case_text, "doctor_reply")
            hospital = self.extract_field(case_text, "hospital")
            department = self.extract_field(case_text, "department")

            # 提取URL（如果有）
            url_match = re.search(r'https?://[^\s]+', case_text)
            url = url_match.group(0) if url_match else ""

            # 创建病例对象
            case = MedicalCase(
                id=case_id,
                year=year,
                disease=disease,
                symptoms=symptoms,
                doctor_reply=doctor_reply,
                hospital=hospital,
                department=department,
                url=url,
                raw_text=case_text[:1000]  # 保存部分原始文本
            )

            # 验证病例
            if self._validate_case(case):
                return case
            else:
                logger.debug(f"病例验证失败: {case_id}")
                self.stats["failed_cases"] += 1
                return None

        except Exception as e:
            logger.debug(f"解析病例失败: {e}")
            self.stats["failed_cases"] += 1
            return None

    def _validate_case(self, case: MedicalCase) -> bool:
        """验证病例是否有效"""
        # 必须有疾病或症状或医生建议
        if not case.disease and not case.symptoms and not case.doctor_reply:
            return False

        # 完整文本不能太短
        if len(case.full_text.strip()) < 50:
            return False

        # 检查是否包含有用的医疗信息
        medical_keywords = ["病", "症", "痛", "疗", "药", "医", "院", "科", "检查", "诊断"]
        text_lower = case.full_text.lower()

        has_medical_content = any(keyword in text_lower for keyword in medical_keywords)
        if not has_medical_content:
            return False

        return True

    def process_single_file(self, file_path: Path, max_cases: Optional[int] = None) -> List[MedicalCase]:
        """处理单个文本文件"""
        year = self.extract_year_from_filename(file_path.name)
        if year is None:
            logger.warning(f"无法从文件名提取年份: {file_path.name}")
            year = 0

        logger.info(f"开始处理文件: {file_path.name} (年份: {year})")

        # 读取文件
        content = self.read_text_file(file_path)
        if content is None:
            return []

        # 分割病例
        case_texts = self.split_into_cases(content)

        # 解析病例
        cases = []
        for i, case_text in enumerate(case_texts):
            if max_cases and len(cases) >= max_cases:
                break

            case = self.parse_single_case(case_text, year, i)
            if case:
                cases.append(case)

            # 进度报告
            if (i + 1) % 1000 == 0:
                logger.info(f"  已解析 {i + 1}/{len(case_texts)} 个病例")

        logger.info(f"文件 {file_path.name}: 解析了 {len(cases)} 个有效病例")

        # 更新统计
        self.stats["total_cases"] += len(cases)
        self.stats["processed_years"].add(year)

        return cases

    def process_all_files(self,
                         max_files: Optional[int] = None,
                         max_cases_per_file: Optional[int] = None,
                         test_mode: bool = False) -> List[MedicalCase]:
        """处理所有文本文件"""

        # 查找所有文本文件
        text_files = list(self.data_dir.glob("*.txt"))
        if not text_files:
            text_files = list(self.data_dir.glob("*.*"))  # 尝试所有文件

        logger.info(f"找到 {len(text_files)} 个文本文件")

        if test_mode:
            text_files = text_files[:3]
            max_cases_per_file = min(max_cases_per_file or 1000, 1000)
            logger.info(f"测试模式: 处理前 {len(text_files)} 个文件")

        all_cases = []

        for i, file_path in enumerate(text_files):
            if max_files and i >= max_files:
                break

            try:
                cases = self.process_single_file(file_path, max_cases_per_file)
                all_cases.extend(cases)

                # 保存每个文件的处理结果
                if cases:
                    self._save_cases_by_year(cases, file_path.stem)

                logger.info(f"进度: {i+1}/{len(text_files)}，累计病例: {len(all_cases)}")

                # 测试模式提前结束
                if test_mode and len(all_cases) >= 1000:
                    logger.info(f"测试模式达到 {len(all_cases)} 个病例，停止处理")
                    break

            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {e}")
                continue

        # 保存完整数据
        if all_cases:
            self._save_all_cases(all_cases)

        # 打印统计信息
        self._print_statistics()

        return all_cases

    def _save_cases_by_year(self, cases: List[MedicalCase], file_tag: str):
        """按年份保存病例"""
        year_groups = {}
        for case in cases:
            if case.year not in year_groups:
                year_groups[case.year] = []
            year_groups[case.year].append(case)

        for year, year_cases in year_groups.items():
            save_path = self.processed_dir / f"cases_{year}_{file_tag}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(year_cases, f)

            logger.debug(f"保存 {year} 年 {len(year_cases)} 个病例到 {save_path}")

    def _save_all_cases(self, cases: List[MedicalCase]):
        """保存所有病例"""
        save_path = self.processed_dir / "all_cases.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(cases, f)

        # 同时保存为JSON（便于查看）
        json_path = self.processed_dir / "all_cases_sample.json"
        sample_cases = [case.to_dict() for case in cases[:100]]  # 保存前100个作为样本
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample_cases, f, ensure_ascii=False, indent=2)

        logger.info(f"保存所有 {len(cases)} 个病例到 {save_path}")

    def _print_statistics(self):
        """打印处理统计信息"""
        end_time = datetime.now()
        processing_time = (end_time - self.stats["start_time"]).total_seconds()

        print("\n" + "="*60)
        print("📊 数据处理统计")
        print("="*60)
        print(f"总文件数: {self.stats['total_files']}")
        print(f"总病例数: {self.stats['total_cases']}")
        print(f"失败病例: {self.stats['failed_cases']}")
        print(f"处理年份: {sorted(self.stats['processed_years'])}")
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"平均速度: {self.stats['total_cases']/processing_time:.2f} 病例/秒" if processing_time > 0 else "N/A")
        print("="*60)

    def load_processed_cases(self, years: Optional[List[int]] = None) -> List[MedicalCase]:
        """加载已处理的病例"""
        all_cases = []

        # 查找所有pkl文件
        pkl_files = list(self.processed_dir.glob("*.pkl"))

        if not pkl_files:
            logger.warning("没有找到已处理的病例文件")
            return []

        for pkl_file in pkl_files:
            try:
                # 从文件名提取年份
                year_match = re.search(r'cases_(\d+)', pkl_file.name)
                if year_match:
                    file_year = int(year_match.group(1))

                    # 如果指定了年份，只加载指定年份
                    if years and file_year not in years:
                        continue

                    with open(pkl_file, 'rb') as f:
                        cases = pickle.load(f)

                    all_cases.extend(cases)
                    logger.info(f"加载 {pkl_file.name}: {len(cases)} 个病例")

            except Exception as e:
                logger.error(f"加载文件 {pkl_file} 失败: {e}")

        logger.info(f"总共加载 {len(all_cases)} 个病例")
        return all_cases

class DocumentSplitter:
    """文档分割器"""

    def __init__(self, config: Dict = None):
        self.config = config or CHUNKING_CONFIG
        self.splitter = None
        self._init_splitter()

    def _init_splitter(self):
        """初始化文本分割器"""
        strategy = self.config.get("strategy", "recursive")

        if strategy == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"],
                length_function=len,
                separators=self.config["separators"],
                keep_separator=True,
                is_separator_regex=False
            )
        else:
            # 默认使用递归分割
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=128,
                separators=["\n\n", "\n", "。", "；", "，", " ", ""]
            )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        if not documents:
            return []

        splits = self.splitter.split_documents(documents)

        # 过滤太短的块
        filtered_splits = []
        for split in splits:
            if len(split.page_content.strip()) >= self.config.get("min_chunk_size", 50):
                filtered_splits.append(split)

        logger.info(f"文档分割: {len(documents)} -> {len(filtered_splits)} 个块")
        return filtered_splits

    def split_cases(self, cases: List[MedicalCase]) -> List[Document]:
        """分割病例为文档块"""
        # 转换为文档
        documents = [case.to_document() for case in cases if case.full_text]

        # 分割文档
        return self.split_documents(documents)

class FinetuneDataGenerator:
    """微调数据生成器"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 微调数据配置
        self.config = {
            "max_samples": 50000,
            "train_ratio": 0.9,
            "instruction_template": "基于以下医疗病例信息回答问题：\n{context}\n\n问题：{question}"
        }

    def generate_from_cases(self, cases: List[MedicalCase]) -> Dict[str, Any]:
        """从病例生成微调数据"""
        logger.info("开始生成微调数据...")

        training_pairs = []

        for case in cases:
            # 过滤无效病例
            if not case.doctor_reply or len(case.doctor_reply) < 20:
                continue

            # 生成问题（从症状或疾病生成）
            question = self._generate_question(case)
            if not question:
                continue

            # 生成上下文
            context = self._generate_context(case)

            # 生成指令-回答对
            instruction = self.config["instruction_template"].format(
                context=context,
                question=question
            )

            training_pairs.append({
                "instruction": instruction,
                "input": "",
                "output": case.doctor_reply,
                "source": case.id,
                "disease": case.disease,
                "year": case.year
            })

            # 限制数量
            if len(training_pairs) >= self.config["max_samples"]:
                break

        logger.info(f"生成了 {len(training_pairs)} 个训练对")

        # 分割训练/验证集
        split_idx = int(len(training_pairs) * self.config["train_ratio"])
        train_data = training_pairs[:split_idx]
        val_data = training_pairs[split_idx:]

        # 保存数据
        self._save_data(train_data, val_data)

        return {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "output_dir": str(self.output_dir)
        }

    def _generate_question(self, case: MedicalCase) -> str:
        """生成问题"""
        # 优先从症状生成问题
        if case.symptoms and len(case.symptoms) > 10:
            # 提取症状关键词
            symptoms = case.symptoms[:100]  # 限制长度
            return f"出现{symptoms}应该怎么办？"

        # 从疾病生成问题
        elif case.disease:
            return f"患有{case.disease}需要注意什么？"

        # 通用问题
        elif case.doctor_reply:
            return "根据我的情况，医生有什么建议？"

        return ""

    def _generate_context(self, case: MedicalCase) -> str:
        """生成上下文"""
        context_parts = []

        if case.disease:
            context_parts.append(f"疾病：{case.disease}")

        if case.symptoms:
            context_parts.append(f"症状：{case.symptoms[:200]}")  # 限制长度

        if case.hospital:
            context_parts.append(f"医院：{case.hospital}")

        if case.department:
            context_parts.append(f"科室：{case.department}")

        return "\n".join(context_parts) if context_parts else "患者医疗咨询"

    def _save_data(self, train_data: List[Dict], val_data: List[Dict]):
        """保存数据"""
        # 保存为JSONL格式
        train_path = self.output_dir / "train.jsonl"
        val_path = self.output_dir / "val.jsonl"

        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 同时保存为JSON（便于查看）
        sample_path = self.output_dir / "sample_data.json"
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(train_data[:10] + val_data[:10], f, ensure_ascii=False, indent=2)

        logger.info(f"微调数据已保存: {len(train_data)} 训练, {len(val_data)} 验证")

def process_medical_texts(
    data_dir: Optional[str] = None,
    max_files: Optional[int] = None,
    max_cases_per_file: Optional[int] = None,
    test_mode: bool = False,
    rebuild: bool = False
) -> Tuple[List[MedicalCase], List[Document]]:
    """
    处理医疗文本的主函数

    Args:
        data_dir: 数据目录路径
        max_files: 最大处理文件数
        max_cases_per_file: 每个文件最大病例数
        test_mode: 测试模式
        rebuild: 重新构建

    Returns:
        (病例列表, 文档块列表)
    """

    processor = TextFileProcessor(data_dir or str(DATA_DIR))
    splitter = DocumentSplitter()

    # 检查是否已有处理结果
    if not rebuild:
        loaded_cases = processor.load_processed_cases()
        if loaded_cases:
            logger.info(f"加载已处理的 {len(loaded_cases)} 个病例")

            # 分割为文档
            documents = splitter.split_cases(loaded_cases)
            return loaded_cases, documents

    # 处理文本文件
    cases = processor.process_all_files(
        max_files=max_files,
        max_cases_per_file=max_cases_per_file,
        test_mode=test_mode
    )

    # 分割为文档
    documents = splitter.split_cases(cases)

    return cases, documents

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗文本数据处理")
    parser.add_argument("--data-dir", type=str, help="数据目录")
    parser.add_argument("--max-files", type=int, help="最大处理文件数")
    parser.add_argument("--max-cases", type=int, default=200000, help="每个文件最大病例数")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--rebuild", action="store_true", help="重新构建")
    parser.add_argument("--generate-finetune", action="store_true", help="生成微调数据")

    args = parser.parse_args()

    # 处理数据
    cases, documents = process_medical_texts(
        data_dir=args.data_dir,
        max_files=args.max_files,
        max_cases_per_file=args.max_cases,
        test_mode=args.test,
        rebuild=args.rebuild
    )

    # 生成微调数据
    if args.generate_finetune and cases:
        finetune_dir = PROCESSED_DIR / "finetune_data"
        generator = FinetuneDataGenerator(finetune_dir)
        result = generator.generate_from_cases(cases)

        print("\n" + "="*60)
        print("✅ 微调数据生成完成")
        print("="*60)
        for key, value in result.items():
            print(f"  {key}: {value}")