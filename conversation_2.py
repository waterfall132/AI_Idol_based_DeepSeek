import asyncio
import pickle
from typing import List, Set, Tuple
import jieba
import pandas as pd
import os
from pathlib import Path


class TextAnalyzer:
    def __init__(self, stopwords_dir: str = './stopwords'):
        """初始化文本分析器"""
        self.stopwords = self._load_stopwords(stopwords_dir)
        self.df = None
        self.segmented_texts = None
        self.column_name = None

    def _load_stopwords(self, stopwords_dir: str) -> Set[str]:
        """加载停用词"""
        stopwords_files = [
            'hit_stopwords.txt',
            'own_stopwords.txt',
            'cn_stopwords.txt',
            'baidu_stopwords.txt',
            'scu_stopwords.txt'
        ]

        all_stopwords = []
        for filename in stopwords_files:
            filepath = os.path.join(stopwords_dir, filename)
            try:
                with open(filepath, encoding='UTF-8') as f:
                    stopwords = [line.strip() for line in f.readlines()]
                    all_stopwords.extend(stopwords)
            except FileNotFoundError:
                print(f"警告: 停用词文件 {filename} 未找到")

        return set(all_stopwords)

    def load_data(self, excel_path: str, column_name: str) -> None:
        """加载Excel数据"""
        self.df = pd.read_excel(excel_path)
        self.column_name = column_name
        self.df[column_name] = self.df[column_name].astype(str)
        self.df[column_name] = self.df[column_name].fillna("")

    def segment_and_save(self, output_file: str) -> None:
        """分词并保存结果"""
        if self.df is None:
            raise ValueError("请先使用load_data()加载数据")

        texts = self.df[self.column_name].tolist()
        self.segmented_texts = []

        for text in texts:
            words = [word for word in jieba.cut(text) if word not in self.stopwords]
            self.segmented_texts.append(words)

        with open(output_file, 'wb') as f:
            pickle.dump(self.segmented_texts, f)

    def load_segmented_texts(self, output_file: str) -> None:
        """从文件加载分词结果"""
        with open(output_file, 'rb') as f:
            self.segmented_texts = pickle.load(f)

    def _keyword_matching(self, query_words: List[str], text_words: List[str]) -> int:
        """计算关键词匹配度"""
        common_words = set(query_words) & set(text_words)
        return len(common_words)

    async def search(self, query: str) -> Tuple[str, int, int]:
        """搜索相似内容"""
        if self.segmented_texts is None:
            raise ValueError("请先加载或生成分词结果")

        query_words = [word for word in jieba.cut(query) if word not in self.stopwords]
        keyword_scores = [self._keyword_matching(query_words, text_words)
                          for text_words in self.segmented_texts]

        max_score = max(keyword_scores)
        max_index = keyword_scores.index(max_score) if max_score > 0 else -1

        if max_index != -1:
            return (self.df[self.column_name].iloc[max_index], max_score, max_index)
        return ("", 0, -1)


async def main():
    # 使用示例
    analyzer = TextAnalyzer()
    excel_path = r"E:\weibo_read-master\weibo_read-master\weibo_赵嘉敏时间胶囊_cleaned.xlsx"
    column_name = "正文内容"
    output_file = "segmented_texts.pkl"

    # 加载数据
    analyzer.load_data(excel_path, column_name)

    # 如果分词结果不存在，进行分词
    if not os.path.exists(output_file):
        analyzer.segment_and_save(output_file)
    else:
        analyzer.load_segmented_texts(output_file)

    # 搜索示例
    query = "韩国"
    content, score, index = await analyzer.search(query)

    if index != -1:
        print(f"\n与 '{query}' 关键词重合度最高的内容:")
        print(f"内容: {content}")
        print(f"重合关键词数量: {score}")
    else:
        print(f"没有找到与 '{query}' 关键词重合的内容。")


if __name__ == "__main__":
    asyncio.run(main())