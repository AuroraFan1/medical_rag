"""
优化的数据处理模块
专为处理大量医疗数据设计，使用本地嵌入模型
"""

import os
import re
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LangChain 相关
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 导入配置
from config import (
    DATA_DIR, VECTOR_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, MAX_CASES_PER_YEAR, BATCH_SIZE, TEST_MODE, RETRIEVAL_K
)


@dataclass
class MedicalCase:
    """医疗病例数据结构"""
    id: str  # 使用字符串ID
    year: int
    disease: str
    symptoms: str
    doctor_reply: str
    hospital: str
    department: str
    url: str = ""
    full_text: str = ""

    def __post_init__(self):
        """初始化后生成完整文本"""
        if not self.full_text:
            self.full_text = self._create_full_text()

    def _create_full_text(self) -> str:
        """创建完整文本"""
        parts = []

        if self.disease:
            parts.append(f"疾病：{self.disease}")

        if self.symptoms:
            parts.append(f"症状描述：{self.symptoms}")

        if self.doctor_reply:
            parts.append(f"医生建议：{self.doctor_reply}")

        if self.hospital and self.department:
            parts.append(f"医院科室：{self.hospital} {self.department}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_document(self) -> Document:
        """转换为LangChain文档"""
        metadata = {
            "id": self.id,
            "disease": self.disease,
            "hospital": self.hospital,
            "department": self.department,
            "year": self.year,
            "url": self.url,
            "source": f"medical_case_{self.id}"
        }

        return Document(
            page_content=self.full_text,
            metadata=metadata
        )


class MedicalDataProcessor:
    """医疗数据处理类"""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

        # 统计数据
        self.stats = {
            "total_cases": 0,
            "processed_years": [],
            "failed_parses": 0
        }

    def parse_medical_dialog(self, content: str, year: int) -> List[MedicalCase]:
        """
        解析医疗对话数据 - 优化版本

        Args:
            content: 原始文本内容
            year: 年份

        Returns:
            病例列表
        """
        cases = []

        # 尝试不同的分割模式
        # 模式1: 按id=分割
        case_blocks = re.split(r'\nid=', content.strip())

        # 如果没有找到id=分割，尝试其他分割方式
        if len(case_blocks) <= 1:
            # 模式2: 按空行分割
            case_blocks = re.split(r'\n\s*\n', content.strip())

        logger.info(f"找到 {len(case_blocks)} 个病例块")

        for block_idx, block in enumerate(case_blocks):
            if not block.strip():
                continue

            # 限制处理数量（用于测试）
            if TEST_MODE and len(cases) >= 1000:
                break

            try:
                # 提取ID - 尝试多种方式
                case_id = self._extract_case_id(block, block_idx, year)

                # 提取疾病
                disease = self._extract_disease(block)

                # 提取症状描述
                symptoms = self._extract_symptoms(block)

                # 提取医生回复
                doctor_reply = self._extract_doctor_reply(block)

                # 提取医院和科室
                hospital, department = self._extract_hospital_department(block)

                # 提取URL
                url = self._extract_url(block)

                # 创建病例
                case = MedicalCase(
                    id=case_id,
                    year=year,
                    disease=disease,
                    symptoms=symptoms,
                    doctor_reply=doctor_reply,
                    hospital=hospital,
                    department=department,
                    url=url
                )

                # 验证病例数据
                if self._validate_case(case):
                    cases.append(case)
                else:
                    logger.debug(f"病例验证失败: {case_id}")
                    self.stats["failed_parses"] += 1

            except Exception as e:
                logger.debug(f"解析病例块失败: {e}")
                self.stats["failed_parses"] += 1
                continue

        return cases

    def _extract_case_id(self, block: str, block_idx: int, year: int) -> str:
        """提取病例ID"""
        # 尝试从文本开头提取数字ID
        id_match = re.match(r'(\d+)', block.strip())
        if id_match:
            return f"{year}_{id_match.group(1)}"

        # 如果找不到，使用哈希值作为ID
        block_hash = hashlib.md5(block.encode()).hexdigest()[:8]
        return f"{year}_{block_idx}_{block_hash}"

    def _extract_disease(self, block: str) -> str:
        """提取疾病名称"""
        # 尝试多种模式
        patterns = [
            r'疾病\s*[：:]\s*(.*?)\n',
            r'病情\s*[：:]\s*(.*?)\n',
            r'诊断\s*[：:]\s*(.*?)\n',
            r'Disease\s*:\s*(.*?)\n',
        ]

        for pattern in patterns:
            match = re.search(pattern, block, re.IGNORECASE)
            if match:
                disease = match.group(1).strip()
                if disease and len(disease) < 100:  # 避免过长
                    return disease

        # 如果找不到，返回空字符串
        return ""

    def _extract_symptoms(self, block: str) -> str:
        """提取症状描述"""
        # 查找描述部分
        desc_patterns = [
            r'Description\s*\n(.*?)(?:\nDialogue|\nDoctor|\n$|\n\d)',
            r'病情描述\s*[：:]\s*(.*?)(?:\n医生|\n$|\n\d)',
            r'症状\s*[：:]\s*(.*?)(?:\n医生|\n$|\n\d)',
        ]

        for pattern in desc_patterns:
            match = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
            if match:
                symptoms = match.group(1).strip()
                # 清理文本
                symptoms = re.sub(r'\s+', ' ', symptoms)
                return symptoms[:500]  # 限制长度

        return ""

    def _extract_doctor_reply(self, block: str) -> str:
        """提取医生回复"""
        # 查找对话部分
        dialogue_patterns = [
            r'Dialogue\s*\n(.*?)$',
            r'医生回复\s*[：:]\s*(.*?)$',
            r'Doctor.*?\n(.*?)$',
        ]

        for pattern in dialogue_patterns:
            match = re.search(pattern, block, re.DOTALL | re.IGNORECASE)
            if match:
                reply = match.group(1).strip()
                # 清理文本
                reply = re.sub(r'\s+', ' ', reply)
                return reply[:1000]  # 限制长度

        return ""

    def _extract_hospital_department(self, block: str) -> Tuple[str, str]:
        """提取医院和科室"""
        # 查找医生信息
        doctor_patterns = [
            r'Doctor faculty\s*\n(.*?)\n',
            r'医生\s*[：:]\s*(.*?)\n',
            r'医院\s*[：:]\s*(.*?)\n',
        ]

        for pattern in doctor_patterns:
            match = re.search(pattern, block, re.IGNORECASE)
            if match:
                info = match.group(1).strip()
                parts = info.split()
                if len(parts) >= 2:
                    return parts[0], " ".join(parts[1:])
                elif len(parts) == 1:
                    return parts[0], ""

        return "", ""

    def _extract_url(self, block: str) -> str:
        """提取URL"""
        url_match = re.search(r'https?://[^\s]+', block)
        return url_match.group(0) if url_match else ""

    def _validate_case(self, case: MedicalCase) -> bool:
        """验证病例数据是否有效"""
        # 必须有疾病或症状或医生回复
        if not case.disease and not case.symptoms and not case.doctor_reply:
            return False

        # 文本不能太短
        if len(case.full_text) < 50:
            return False

        return True

    def process_year_data(self, year: int, max_cases: int = MAX_CASES_PER_YEAR) -> List[MedicalCase]:
        """处理单个年份的数据"""
        file_path = os.path.join(self.data_dir, f"{year}.txt")

        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return []

        try:
            logger.info(f"开始处理 {year} 年数据...")

            # 读取文件
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 解析数据
            cases = self.parse_medical_dialog(content, year)

            # 限制病例数量
            if len(cases) > max_cases:
                cases = cases[:max_cases]
                logger.info(f"限制为前 {max_cases} 个病例")

            logger.info(f"{year} 年: 处理了 {len(cases)} 个病例")

            # 保存处理后的数据
            year_file = os.path.join(self.processed_dir, f"cases_{year}.pkl")
            with open(year_file, 'wb') as f:
                pickle.dump(cases, f)

            # 更新统计数据
            self.stats["total_cases"] += len(cases)
            self.stats["processed_years"].append(year)

            return cases

        except Exception as e:
            logger.error(f"处理 {year} 年数据时出错: {e}")
            return []

    def load_processed_cases(self, years: Optional[List[int]] = None) -> List[MedicalCase]:
        """加载已处理的病例数据"""
        all_cases = []

        # 如果没有指定年份，加载所有已处理的年份
        if years is None:
            # 查找所有pkl文件
            pkl_files = [f for f in os.listdir(self.processed_dir) if f.endswith('.pkl')]
            years = []
            for file in pkl_files:
                try:
                    year = int(re.search(r'cases_(\d+)\.pkl', file).group(1))
                    years.append(year)
                except:
                    continue

        for year in years:
            year_file = os.path.join(self.processed_dir, f"cases_{year}.pkl")
            if os.path.exists(year_file):
                try:
                    with open(year_file, 'rb') as f:
                        cases = pickle.load(f)
                    all_cases.extend(cases)
                    logger.info(f"加载 {year} 年数据: {len(cases)} 个病例")
                except Exception as e:
                    logger.error(f"加载 {year} 年数据失败: {e}")

        return all_cases

    def create_documents(self, cases: List[MedicalCase]) -> List[Document]:
        """将病例转换为文档"""
        documents = []

        for case in cases:
            try:
                doc = case.to_document()
                documents.append(doc)
            except Exception as e:
                logger.debug(f"转换病例失败: {e}")
                continue

        logger.info(f"创建了 {len(documents)} 个文档")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为块"""
        if not documents:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            keep_separator=True
        )

        splits = text_splitter.split_documents(documents)
        logger.info(f"文档分割完成: {len(documents)} -> {len(splits)} 个块")
        return splits


class VectorDatabaseManager:
    """向量数据库管理器"""

    def __init__(self, persist_directory: str = VECTOR_STORE_PATH):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.embeddings = None

    def initialize_embeddings(self, model_name: str = EMBEDDING_MODEL) -> bool:
        """初始化嵌入模型"""
        try:
            logger.info(f"初始化嵌入模型: {model_name}")

            # 使用本地模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cuda'},  # 使用CPU
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )

            logger.info("嵌入模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {e}")

            # 尝试备用模型
            try:
                logger.info("尝试备用模型: sentence-transformers/all-MiniLM-L6-v2")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cuda'}
                )
                logger.info("备用嵌入模型初始化成功")
                return True
            except Exception as e2:
                logger.error(f"备用模型也失败: {e2}")
                return False

    def create_vectorstore(self, documents: List[Document], batch_size: int = BATCH_SIZE) -> bool:
        """创建向量数据库"""
        if not documents:
            logger.error("没有文档可用于创建向量存储")
            return False

        if self.embeddings is None:
            logger.error("嵌入模型未初始化")
            return False

        try:
            logger.info(f"开始创建向量数据库，共 {len(documents)} 个文档...")

            # 创建临时目录
            temp_dir = self.persist_directory + "_temp"
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

            # 分批处理
            total_docs = len(documents)
            processed = 0

            for i in range(0, total_docs, batch_size):
                batch = documents[i:min(i + batch_size, total_docs)]

                if i == 0:
                    # 创建第一个批次
                    self.vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        persist_directory=temp_dir,
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                else:
                    # 添加后续批次
                    self.vectorstore.add_documents(batch)

                processed += len(batch)
                progress = (processed / total_docs) * 100

                if (i // batch_size) % 10 == 0 or i + batch_size >= total_docs:
                    logger.info(f"  进度: {processed}/{total_docs} ({progress:.1f}%)")

            # 持久化
            self.vectorstore.persist()

            import shutil
            # 移动到最终位置
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory)
            shutil.move(temp_dir, self.persist_directory)

            logger.info(f"向量数据库创建完成: {self.persist_directory}")

            # 验证
            count = self.vectorstore._collection.count()
            logger.info(f"向量数据库包含 {count} 个向量")

            return True

        except Exception as e:
            logger.error(f"创建向量数据库失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def load_vectorstore(self) -> bool:
        """加载现有向量数据库"""
        if not os.path.exists(self.persist_directory):
            logger.warning(f"向量数据库不存在: {self.persist_directory}")
            return False

        try:
            logger.info(f"加载向量数据库: {self.persist_directory}")

            # 确保嵌入模型已初始化
            if self.embeddings is None:
                self.initialize_embeddings()

            # 加载向量数据库
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

            # 验证
            count = self.vectorstore._collection.count()
            logger.info(f"向量数据库加载成功，包含 {count} 个向量")

            return True

        except Exception as e:
            logger.error(f"加载向量数据库失败: {e}")
            return False

    def search_similar(self, query: str, k: int = RETRIEVAL_K) -> List[Document]:
        """搜索相似文档"""
        if self.vectorstore is None:
            logger.error("向量数据库未加载")
            return []

        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_info(self) -> Dict[str, Any]:
        """获取向量数据库信息"""
        if self.vectorstore is None:
            return {"status": "未初始化"}

        try:
            count = self.vectorstore._collection.count()
            return {
                "status": "已加载",
                "向量数量": count,
                "存储路径": self.persist_directory
            }
        except:
            return {"status": "未知"}


def build_medical_vector_database(
    years: Optional[List[int]] = None,
    max_cases_per_year: int = MAX_CASES_PER_YEAR,
    rebuild: bool = False,
    test_mode: bool = False
) -> Optional[VectorDatabaseManager]:
    """
    构建医疗向量数据库的主函数

    Args:
        years: 要处理的年份列表
        max_cases_per_year: 每个年份最大病例数
        rebuild: 是否重新构建
        test_mode: 测试模式，只处理少量数据

    Returns:
        VectorDatabaseManager实例或None
    """

    # 如果没有指定年份，使用默认年份
    if years is None:
        years = list(range(2010, 2021))  # 2010-2020

    logger.info("开始构建医疗向量数据库...")
    logger.info(f"处理年份: {years}")
    logger.info(f"每年来最大病例数: {max_cases_per_year}")
    logger.info(f"测试模式: {test_mode}")

    # 0. 检查是否已存在
    manager = VectorDatabaseManager()

    if not rebuild and os.path.exists(VECTOR_STORE_PATH):
        logger.info("向量数据库已存在，尝试加载...")
        if manager.load_vectorstore():
            logger.info("✅ 向量数据库加载成功")
            return manager
        else:
            logger.warning("加载失败，将重新构建")

    # 1. 初始化嵌入模型
    logger.info("初始化嵌入模型...")
    if not manager.initialize_embeddings():
        logger.error("无法初始化嵌入模型")
        return None

    # 2. 处理数据
    processor = MedicalDataProcessor()
    all_cases = []

    for year in years:
        # 测试模式只处理一个年份
        if test_mode and len(all_cases) > 1000:
            logger.info(f"测试模式，已达到 {len(all_cases)} 个病例，停止处理")
            break

        cases = processor.process_year_data(year, max_cases_per_year)
        all_cases.extend(cases)

    logger.info(f"总共处理了 {len(all_cases)} 个病例")

    # 3. 转换为文档
    logger.info("转换为文档...")
    documents = processor.create_documents(all_cases)

    # 4. 分割文档
    logger.info("分割文档...")
    splits = processor.split_documents(documents)

    # 5. 测试模式减少数据量
    if test_mode:
        test_size = min(10000, len(splits))
        splits = splits[:test_size]
        logger.info(f"测试模式: 使用 {test_size} 个文档块")

    # 6. 创建向量数据库
    logger.info("创建向量数据库...")
    success = manager.create_vectorstore(splits)

    if success:
        logger.info("✅ 向量数据库构建成功")
        return manager
    else:
        logger.error("❌ 向量数据库构建失败")
        return None


if __name__ == "__main__":
    """直接运行此文件以构建向量数据库"""
    import argparse

    parser = argparse.ArgumentParser(description="构建医疗向量数据库")
    parser.add_argument("--rebuild", action="store_true", help="强制重新构建")
    parser.add_argument("--test", action="store_true", help="测试模式（使用少量数据）")
    parser.add_argument("--years", type=str, help="指定年份，用逗号分隔，如：2010,2011,2012")
    parser.add_argument("--max-cases", type=int, default=MAX_CASES_PER_YEAR,
                       help=f"每年来最大病例数，默认: {MAX_CASES_PER_YEAR}")

    args = parser.parse_args()

    # 解析年份参数
    years = None
    if args.years:
        try:
            years = [int(y.strip()) for y in args.years.split(",")]
        except:
            logger.error("年份格式错误，请使用逗号分隔的数字")
            exit(1)

    # 构建向量数据库
    result = build_medical_vector_database(
        years=years,
        max_cases_per_year=args.max_cases,
        rebuild=args.rebuild,
        test_mode=args.test
    )

    if result:
        # 显示信息
        info = result.get_info()
        print("\n" + "="*50)
        print("✅ 向量数据库构建成功！")
        print("="*50)
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\n🎯 接下来可以运行 Streamlit 应用：")
        print("streamlit run streamlit_app.py")
    else:
        print("\n❌ 构建失败，请查看上面的错误信息")