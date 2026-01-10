"""
从.pkl文件创建测试数据
"""

import json
import random
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PKLTestDataCreator:
    """从.pkl文件创建测试数据"""

    def __init__(self, processed_dir: str = "./processed"):
        self.processed_dir = Path(processed_dir)
        self.test_data = []

    def find_pkl_files(self) -> List[Path]:
        """查找.pkl文件"""
        pkl_files = list(self.processed_dir.glob("*.pkl"))
        logger.info(f"找到 {len(pkl_files)} 个.pkl文件")
        return pkl_files

    def load_cases_from_pkl(self, pkl_file: Path, limit: int = 1000) -> List:
        """从.pkl文件加载病例数据"""
        try:
            logger.info(f"加载病例文件: {pkl_file.name}")
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # 检查数据类型
            if isinstance(data, list):
                cases = data
                logger.info(f"加载了 {len(cases)} 个病例")
                return cases[:limit] if limit else cases
            else:
                logger.warning(f"文件 {pkl_file.name} 的数据不是列表格式")
                return []

        except Exception as e:
            logger.error(f"加载 {pkl_file} 失败: {e}")
            return []

    def extract_qa_from_case(self, case) -> Dict:
        """从单个病例中提取QA对"""

        # 尝试不同的属性访问方式
        question = ""
        answer = ""
        disease = ""

        try:
            # 方式1: 如果是字典
            if isinstance(case, dict):
                question = case.get('patient_desc', case.get('symptoms', case.get('description', '')))
                answer = case.get('doctor_reply', case.get('advice', case.get('response', '')))
                disease = case.get('disease', '')

            # 方式2: 如果是对象（有属性）
            elif hasattr(case, 'patient_desc'):
                question = getattr(case, 'patient_desc', '')
                answer = getattr(case, 'doctor_reply', '')
                disease = getattr(case, 'disease', '')

            # 方式3: 尝试从对话中提取
            if (not question or not answer) and hasattr(case, 'dialogue'):
                dialogue = getattr(case, 'dialogue', '')
                qa = self._extract_from_dialogue(dialogue)
                if qa:
                    question = qa.get('question', question)
                    answer = qa.get('answer', answer)

        except Exception as e:
            logger.warning(f"提取病例数据失败: {e}")

        # 清理文本
        question = self._clean_text(question)
        answer = self._clean_text(answer)

        # 如果还是没有问答，尝试从症状和诊断中生成
        if not question or not answer:
            qa = self._generate_qa_from_case(case)
            if qa:
                question = qa.get('question', question)
                answer = qa.get('answer', answer)

        if question and answer and len(answer) > 20:
            return {
                'question': question,
                'answer': answer,
                'disease': disease,
                'source': 'pkl_cases',
                'original_length': len(answer)
            }

        return None

    def _extract_from_dialogue(self, dialogue: str) -> Dict:
        """从对话文本中提取QA"""
        if not dialogue:
            return None

        patterns = [
            (r'患者[：:]\s*(.*?)(?=\n医生|\n$)', r'医生[：:]\s*(.*?)(?=\n患者|\n$)'),
            (r'病人[：:]\s*(.*?)(?=\n医生|\n$)', r'医生[：:]\s*(.*?)(?=\n病人|\n$)'),
            (r'问[：:]\s*(.*?)(?=\n答|\n$)', r'答[：:]\s*(.*?)(?=\n问|\n$)'),
            (r'病情描述[：:]\s*(.*?)(?=\n医生建议|\n$)', r'医生建议[：:]\s*(.*?)(?=\n$)'),
            (r'症状[：:]\s*(.*?)(?=\n诊断|\n$)', r'诊断[：:]\s*(.*?)(?=\n$)')
        ]

        for q_pattern, a_pattern in patterns:
            q_match = re.search(q_pattern, dialogue, re.DOTALL)
            a_match = re.search(a_pattern, dialogue, re.DOTALL)

            if q_match and a_match:
                return {
                    'question': q_match.group(1).strip(),
                    'answer': a_match.group(1).strip()
                }

        # 尝试简单的换行分割
        lines = [line.strip() for line in dialogue.split('\n') if line.strip()]
        if len(lines) >= 2:
            for i in range(len(lines) - 1):
                if ('患者' in lines[i] or '问' in lines[i]) and ('医生' in lines[i+1] or '答' in lines[i+1]):
                    question = re.sub(r'^(患者|病人|问)[：:]\s*', '', lines[i])
                    answer = re.sub(r'^(医生|答)[：:]\s*', '', lines[i+1])
                    return {'question': question, 'answer': answer}

        return None

    def _generate_qa_from_case(self, case) -> Dict:
        """从病例信息生成QA"""
        try:
            # 尝试获取疾病信息
            if isinstance(case, dict):
                disease = case.get('disease', '')
                symptoms = case.get('symptoms', '')
            elif hasattr(case, 'disease'):
                disease = getattr(case, 'disease', '')
                symptoms = getattr(case, 'symptoms', '')
            else:
                disease = ''
                symptoms = ''

            if disease and symptoms:
                # 生成问题
                question_templates = [
                    f"患有{disease}，症状是{symptoms}，应该怎么办？",
                    f"{disease}的症状是{symptoms}，如何治疗？",
                    f"得了{disease}，有{symptoms}，要注意什么？"
                ]

                # 生成答案
                answer_templates = {
                    '糖尿病': "糖尿病患者应注意低糖饮食，控制碳水化合物摄入，定期监测血糖。",
                    '高血压': "高血压患者应低盐饮食，规律服药，定期监测血压，保持健康生活方式。",
                    '感冒': "感冒患者应多休息，多喝水，必要时服用对症药物。",
                    '胃炎': "胃炎患者应注意饮食清淡，避免刺激性食物，必要时服用胃药。",
                    '肺炎': "肺炎患者应及时就医，按医嘱服药，多休息，加强营养。"
                }

                question = random.choice(question_templates)
                answer = answer_templates.get(disease, f"{disease}患者应根据具体症状咨询专业医生，制定个性化治疗方案。")

                return {'question': question, 'answer': answer}

        except Exception as e:
            logger.warning(f"生成QA失败: {e}")

        return None

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""

        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # 移除常见前缀
        prefixes = [
            r'^患者[：:]\s*',
            r'^医生[：:]\s*',
            r'^问[：:]\s*',
            r'^答[：:]\s*',
            r'^病人[：:]\s*',
            r'^病情描述[：:]\s*',
            r'^症状[：:]\s*',
            r'^诊断[：:]\s*'
        ]

        for prefix in prefixes:
            text = re.sub(prefix, '', text)

        return text.strip()

    def create_test_dataset(self,
                           output_path: str = "./data/test_from_pkl.json",
                           max_samples: int = 200) -> Dict[str, Any]:
        """创建测试数据集"""

        # 查找.pkl文件
        pkl_files = self.find_pkl_files()
        if not pkl_files:
            logger.error("没有找到.pkl文件")
            return {}

        all_cases = []
        test_data = []

        # 加载所有病例
        for pkl_file in pkl_files[:11]:  # 只处理前3个文件
            cases = self.load_cases_from_pkl(pkl_file, limit=500)
            all_cases.extend(cases)

            if len(all_cases) >= max_samples * 3:  # 多加载一些用于筛选
                break

        logger.info(f"总共加载了 {len(all_cases)} 个病例")

        # 从病例中提取QA
        for case in all_cases:
            if len(test_data) >= max_samples:
                break

            qa = self.extract_qa_from_case(case)
            if qa:
                test_data.append(qa)

        logger.info(f"创建了 {len(test_data)} 个测试样本")

        # 如果没有足够的样本，生成一些补充数据
        if len(test_data) < 50:
            logger.info("样本不足，生成补充数据...")
            supplementary_data = self._create_supplementary_data(50 - len(test_data))
            test_data.extend(supplementary_data)

        # 打乱顺序
        random.shuffle(test_data)

        # 保存到文件
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        # 统计信息
        stats = self._analyze_test_data(test_data)

        # 保存统计信息
        stats_file = output_path_obj.parent / f"{output_path_obj.stem}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"测试数据集已保存: {output_path}")
        logger.info(f"统计信息: {stats}")

        return {
            'data': test_data,
            'stats': stats,
            'file_path': str(output_path_obj)
        }

    def _create_supplementary_data(self, num_samples: int) -> List[Dict]:
        """创建补充测试数据"""
        supplementary_data = []

        # 常见的医疗QA对
        common_qa_pairs = [
            {
                'question': '糖尿病患者应该注意什么饮食？',
                'answer': '糖尿病患者应注意低糖低脂饮食，控制碳水化合物摄入，多吃蔬菜和全谷物，定期监测血糖。',
                'disease': '糖尿病',
                'source': 'supplementary'
            },
            {
                'question': '高血压怎么治疗？',
                'answer': '高血压治疗包括：1. 非药物治疗（低盐饮食、适量运动、戒烟限酒）2. 药物治疗（降压药）3. 定期监测血压。',
                'disease': '高血压',
                'source': 'supplementary'
            },
            {
                'question': '感冒发烧应该吃什么药？',
                'answer': '感冒发烧可服用布洛芬或对乙酰氨基酚退烧。同时应多喝水、多休息，如果持续高热不退需及时就医。',
                'disease': '感冒',
                'source': 'supplementary'
            },
            {
                'question': '胃痛怎么办？',
                'answer': '胃痛处理：1. 清淡饮食，避免刺激性食物 2. 可服用胃黏膜保护剂 3. 如疼痛剧烈或持续不缓解需及时就医。',
                'disease': '胃炎',
                'source': 'supplementary'
            },
            {
                'question': '头痛怎么缓解？',
                'answer': '头痛缓解方法：1. 休息 2. 可服用止痛药如布洛芬 3. 热敷或冷敷 4. 如头痛剧烈或反复发作需就医检查。',
                'disease': '头痛',
                'source': 'supplementary'
            },
            {
                'question': '咳嗽应该吃什么药？',
                'answer': '咳嗽治疗：1. 干咳可用右美沙芬 2. 有痰可用氨溴索或乙酰半胱氨酸 3. 如持续咳嗽2周以上需就医检查。',
                'disease': '咳嗽',
                'source': 'supplementary'
            },
            {
                'question': '失眠怎么办？',
                'answer': '失眠处理方法：1. 建立规律作息 2. 睡前避免咖啡、浓茶 3. 可短期使用助眠药物 4. 如长期失眠需就医。',
                'disease': '失眠',
                'source': 'supplementary'
            },
            {
                'question': '腹泻应该吃什么药？',
                'answer': '腹泻处理：1. 可服用蒙脱石散止泻 2. 补充口服补液盐防止脱水 3. 如伴有发热或血便需及时就医。',
                'disease': '腹泻',
                'source': 'supplementary'
            }
        ]

        # 复制已有的QA对
        while len(supplementary_data) < num_samples:
            qa = random.choice(common_qa_pairs)
            supplementary_data.append(qa.copy())

            if len(supplementary_data) >= num_samples:
                break

        return supplementary_data[:num_samples]

    def _analyze_test_data(self, test_data: List[Dict]) -> Dict[str, Any]:
        """分析测试数据"""

        stats = {
            'total': len(test_data),
            'question_lengths': [],
            'answer_lengths': [],
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'diseases': {},
            'sources': {}
        }

        disease_counts = {}
        source_counts = {}
        total_q_len = 0
        total_a_len = 0

        for item in test_data:
            # 长度统计
            q_len = len(item['question'])
            a_len = len(item['answer'])
            stats['question_lengths'].append(q_len)
            stats['answer_lengths'].append(a_len)
            total_q_len += q_len
            total_a_len += a_len

            # 疾病统计
            disease = item.get('disease', '未知')
            disease_counts[disease] = disease_counts.get(disease, 0) + 1

            # 来源统计
            source = item.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        stats['avg_question_length'] = total_q_len / len(test_data) if test_data else 0
        stats['avg_answer_length'] = total_a_len / len(test_data) if test_data else 0
        stats['diseases'] = disease_counts
        stats['sources'] = source_counts

        return stats

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="从.pkl文件创建测试数据集")
    parser.add_argument("--processed-dir", type=str, default="./processed", help="已处理数据目录")
    parser.add_argument("--output", type=str, default="./data/test_from_pkl.json", help="输出文件路径")
    parser.add_argument("--max-samples", type=int, default=5000, help="最大样本数")

    args = parser.parse_args()

    print("="*60)
    print("从.pkl文件创建测试数据集")
    print("="*60)

    creator = PKLTestDataCreator(args.processed_dir)
    result = creator.create_test_dataset(
        output_path=args.output,
        max_samples=args.max_samples
    )

    if result:
        print("\n✅ 测试数据集创建成功！")
        print(f"文件位置: {result['file_path']}")

        stats = result['stats']
        print(f"\n📊 数据集统计:")
        print(f"总问题数: {stats['total']}")
        print(f"平均问题长度: {stats['avg_question_length']:.1f} 字符")
        print(f"平均答案长度: {stats['avg_answer_length']:.1f} 字符")

        print("\n疾病分布:")
        for disease, count in stats['diseases'].items():
            print(f"  {disease}: {count}")

        print("\n来源分布:")
        for source, count in stats['sources'].items():
            print(f"  {source}: {count}")
    else:
        print("\n❌ 创建测试数据集失败")

if __name__ == "__main__":
    main()