"""
数据处理和向量数据库构建模块
处理原始医疗对话数据，构建并保存向量数据库
"""

import os
import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle

# LangChain 相关
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from chromadb.config import Settings

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MedicalCase:
    """医疗病例数据结构"""
    id: int
    url: str
    disease: str
    symptoms: str
    medical_history: str
    help_request: str
    doctor_reply: str
    hospital: str
    department: str
    year: int
    full_text: str


class DataProcessor:
    """数据处理类，专门处理原始医疗对话数据"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

    def parse_medical_dialog(self, content: str, year: int) -> List[MedicalCase]:
        """
        解析医疗对话数据
        """
        cases = []

        # 分割每个病例块
        case_blocks = re.split(r'\nid=', content.strip())

        for block in case_blocks:
            if not block.strip():
                continue

            try:
                # 提取ID
                id_match = re.match(r'(\d+)', block)
                if not id_match:
                    continue

                case_id = int(id_match.group(1))

                # 提取URL
                url_match = re.search(r'https?://[^\s]+', block)
                url = url_match.group(0) if url_match else ""

                # 提取医生科室
                faculty_match = re.search(r'Doctor faculty\s*\n(.*?)\n', block, re.DOTALL)
                doctor_faculty = faculty_match.group(1).strip() if faculty_match else ""

                # 提取医院和科室
                hospital, department = "", ""
                if doctor_faculty:
                    parts = doctor_faculty.split()
                    if len(parts) >= 2:
                        hospital = parts[0]
                        department = " ".join(parts[1:])

                # 提取疾病
                disease_match = re.search(r'疾病：\s*\n(.*?)\n', block)
                disease = disease_match.group(1).strip() if disease_match else ""

                # 提取描述部分
                description_text = ""
                desc_match = re.search(r'Description\s*\n(.*?)\nDialogue', block, re.DOTALL)
                if desc_match:
                    desc_text = desc_match.group(1)
                    description_text = desc_text.strip()

                # 提取对话
                dialogue_text = ""
                dialogue_match = re.search(r'Dialogue\s*\n(.*?)$', block, re.DOTALL)
                if dialogue_match:
                    dialogue_text = dialogue_match.group(1).strip()

                # 创建完整文本
                full_text = f"疾病：{disease}\n"
                if description_text:
                    full_text += f"病情描述：{description_text}\n"
                if dialogue_text:
                    full_text += f"医患对话：{dialogue_text}"

                # 创建MedicalCase对象
                case = MedicalCase(
                    id=case_id,
                    url=url,
                    disease=disease,
                    symptoms=description_text,
                    medical_history="",
                    help_request="",
                    doctor_reply=dialogue_text,
                    hospital=hospital,
                    department=department,
                    year=year,
                    full_text=full_text
                )

                cases.append(case)

            except Exception as e:
                logger.error(f"解析病例 {block[:100]}... 时出错: {e}")
                continue

        return cases

    def process_all_data(self, years: List[int] = None) -> List[MedicalCase]:
        """
        处理所有年份的数据
        """
        if years is None:
            years = range(2010,2026)  # 默认处理2011年数据

        all_cases = []

        for year in years:
            file_path = os.path.join(self.data_dir, f"{year}.txt")
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                cases = self.parse_medical_dialog(content, year)
                all_cases.extend(cases)
                logger.info(f"处理 {year} 年数据，共 {len(cases)} 个病例")

            except Exception as e:
                logger.error(f"处理 {year} 年数据时出错: {e}")

        # 保存处理后的数据
        processed_file = os.path.join(self.processed_dir, "medical_cases.pkl")
        with open(processed_file, 'wb') as f:
            pickle.dump(all_cases, f)

        logger.info(f"共处理 {len(all_cases)} 个病例，已保存到 {processed_file}")
        return all_cases

    def create_documents_from_cases(self, cases: List[MedicalCase]) -> List[Document]:
        """
        将病例数据转换为LangChain Document对象
        """
        documents = []

        for case in cases:
            # 创建元数据
            metadata = {
                "id": case.id,
                "disease": case.disease,
                "hospital": case.hospital,
                "department": case.department,
                "year": case.year,
                "url": case.url
            }

            # 创建文档
            doc = Document(
                page_content=case.full_text,
                metadata=metadata
            )
            documents.append(doc)

        return documents

    def split_documents(self, documents: List[Document],
                        chunk_size: int = 500,
                        chunk_overlap: int = 100) -> List[Document]:
        """
        分割文档
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""]
        )

        splits = text_splitter.split_documents(documents)
        logger.info(f"文档分割完成，共 {len(splits)} 个块")
        return splits


class VectorStoreManager:
    """向量存储管理器"""
    def __init__(self, persist_directory: str = "chroma_db_medical"):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.embeddings = None

    def initialize(self, embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """初始化向量存储"""
        try:
            # 检查嵌入模型类型
            if embedding_model.startswith("sentence-transformers/"):
                # 使用 HuggingFace 嵌入模型（本地）
                logger.info(f"使用 HuggingFace 嵌入模型: {embedding_model}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model,
                    model_kwargs={'device': 'cuda'},  # 使用CPU，如果有GPU可以改为'cuda'
                    encode_kwargs={'normalize_embeddings': False}
                )
            else:
                # 使用 OpenAI 兼容的嵌入模型（API）
                logger.info(f"使用 OpenAI 兼容嵌入模型: {embedding_model}")
                self.embeddings = OpenAIEmbeddings(model=embedding_model)

            logger.info(f"嵌入模型初始化完成: {embedding_model}")

            # 检查是否已存在向量数据库
            if os.path.exists(self.persist_directory):
                logger.info(f"加载现有向量数据库: {self.persist_directory}")
                try:
                    self.vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    # 验证向量数据库是否与嵌入模型兼容
                    test_query = "测试"
                    results = self.vectorstore.similarity_search(test_query, k=1)
                    logger.info(f"向量数据库验证成功，包含 {self.vectorstore._collection.count()} 个向量")
                    return True
                except Exception as e:
                    logger.error(f"加载现有向量数据库失败: {e}")
                    logger.info("可能需要重新构建向量数据库")
                    return False
            else:
                logger.info("准备创建新的向量数据库")
                return False

        except Exception as e:
            logger.error(f"初始化向量存储时出错: {e}")
            return False

        # ... 其他方法保持不变 ...


    # def __init__(self, persist_directory: str = "chroma_db_medical"):
    #     self.persist_directory = persist_directory
    #     self.vectorstore = None
    #     self.embeddings = None
    #
    # def initialize(self, embedding_model: str = "BAAI/bge-m3"):
    #     """初始化向量存储"""
    #     try:
    #         # 初始化嵌入模型
    #         self.embeddings = OpenAIEmbeddings(model=embedding_model)
    #         logger.info(f"嵌入模型初始化完成: {embedding_model}")
    #
    #         # 检查是否已存在向量数据库
    #         if os.path.exists(self.persist_directory):
    #             logger.info(f"加载现有向量数据库: {self.persist_directory}")
    #             self.vectorstore = Chroma(
    #                 persist_directory=self.persist_directory,
    #                 embedding_function=self.embeddings
    #             )
    #             return True
    #         else:
    #             logger.info("准备创建新的向量数据库")
    #             return False
    #
    #     except Exception as e:
    #         logger.error(f"初始化向量存储时出错: {e}")
    #         return False

    def create_vectorstore(self, documents: List[Document]) -> bool:
        """
        创建向量存储
        """
        try:
            if not documents:
                logger.error("没有文档可用于创建向量存储")
                return False

            logger.info(f"正在创建向量存储，使用 {len(documents)} 个文档...")

            # 创建向量存储
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

            # 持久化
            self.vectorstore.persist()

            logger.info(f"向量存储创建完成，已保存到 {self.persist_directory}")
            return True

        except Exception as e:
            logger.error(f"创建向量存储时出错: {e}")
            return False

    def get_retriever(self, search_kwargs: dict = None):
        """
        获取检索器
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化")

        if search_kwargs is None:
            search_kwargs = {"k": 5}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def get_vectorstore_info(self):
        """获取向量存储信息"""
        if self.vectorstore is None:
            return {"status": "未初始化"}

        try:
            # ChromaDB获取集合信息
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = client.list_collections()

            info = {
                "status": "已初始化",
                "存储路径": self.persist_directory,
                "集合数量": len(collections)
            }

            for collection in collections:
                count = collection.count()
                info[collection.name] = f"{count} 条记录"

            return info

        except Exception as e:
            logger.error(f"获取向量存储信息时出错: {e}")
            return {"status": "错误", "error": str(e)}


def build_medical_rag_system(force_rebuild: bool = False):
    """
    构建医疗RAG系统的完整流程
    """
    logger.info("开始构建医疗RAG系统...")

    # 1. 初始化组件
    data_processor = DataProcessor()
    vector_manager = VectorStoreManager()

    # 2. 检查向量数据库是否存在
    vector_exists = vector_manager.initialize()

    if vector_exists and not force_rebuild:
        logger.info("向量数据库已存在，跳过重建")
        return vector_manager

    # 3. 处理数据
    logger.info("开始处理原始数据...")
    cases = data_processor.process_all_data()

    if not cases:
        logger.error("没有处理到任何病例数据")
        return None

    # 4. 转换为文档
    documents = data_processor.create_documents_from_cases(cases)

    # 5. 分割文档
    splits = data_processor.split_documents(documents)

    # 6. 创建向量数据库
    success = vector_manager.create_vectorstore(splits)

    if success:
        logger.info("医疗RAG系统构建完成")
        return vector_manager
    else:
        logger.error("构建医疗RAG系统失败")
        return None


def check_vectorstore_compatibility():
    """
    检查向量数据库兼容性
    """
    vector_manager = VectorStoreManager()

    # 尝试不同维度的模型
    test_models = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384维
        "sentence-transformers/all-MiniLM-L6-v2",  # 384维
        "BAAI/bge-m3",  # 1024维
        "text-embedding-ada-002"  # 1536维
    ]

    for model in test_models:
        print(f"\n测试模型: {model}")
        try:
            # 初始化嵌入
            if model.startswith("sentence-transformers/"):
                embeddings = HuggingFaceEmbeddings(
                    model_name=model,
                    model_kwargs={'device': 'cuda'}
                )
            else:
                embeddings = OpenAIEmbeddings(model=model)

            # 尝试加载向量数据库
            vectorstore = Chroma(
                persist_directory="chroma_db_medical",
                embedding_function=embeddings
            )

            # 测试查询
            results = vectorstore.similarity_search("测试", k=1)
            print(f"✓ 兼容！向量数量: {vectorstore._collection.count()}")
            return model

        except Exception as e:
            print(f"✗ 不兼容: {e}")

    return None


if __name__ == "__main__":
    # 直接运行此文件可以构建向量数据库或检查兼容性
    import argparse

    parser = argparse.ArgumentParser(description="构建医疗RAG向量数据库")
    parser.add_argument("--rebuild", action="store_true", help="强制重新构建向量数据库")
    parser.add_argument("--check", action="store_true", help="检查向量数据库兼容性")
    args = parser.parse_args()

    if args.check:
        # 检查兼容性
        compatible_model = check_vectorstore_compatibility()
        if compatible_model:
            print(f"\n推荐使用的模型: {compatible_model}")
        else:
            print("\n没有找到兼容的模型，需要重新构建向量数据库")
    else:
        # 构建系统
        result = build_medical_rag_system(force_rebuild=args.rebuild)

        if result:
            # 显示系统信息
            info = result.get_vectorstore_info()
            print("\n向量数据库信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print("构建失败，请检查日志")