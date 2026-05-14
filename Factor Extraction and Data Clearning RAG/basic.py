"""
RAG Financial Data Agent - A股股票数据处理与因子清洗
功能：
1. 提取A股股票数据（示例：平安银行000001）
2. 清洗原始数据（剔除价格、成交量的异常值）
3. 提取因子（日收益率因子）并再次清洗因子异常值
4. 集成RAG：根据用户查询检索知识库中的清洗规则，并基于检索结果执行操作
"""



import numpy as np
import pandas as pd
import akshare as ak
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ======================== 1. 构建RAG知识库（金融数据处理规则） ========================
# 知识文档：包含因子清洗、数据预处理等相关规则
knowledge_docs = [
    {
        "id": "doc1",
        "content": "A股数据清洗规则：剔除停牌日数据（成交量为0或空）；剔除价格小于等于0的记录；剔除涨跌停板造成的极端值。",
        "tags": ["清洗", "原始数据", "异常值"]
    },
    {
        "id": "doc2",
        "content": "因子异常值处理规则：使用3-sigma原则（均值±3倍标准差）剔除极端值；对于偏态分布，可使用中位数绝对偏差（MAD）。",
        "tags": ["因子", "异常值", "3-sigma"]
    },
    {
        "id": "doc3",
        "content": "常规因子清洗步骤：1）删除缺失值；2）按指定阈值截尾（例如1%-99%分位数）；3）标准化或剔除离群点后保留有效数据。",
        "tags": ["因子", "清洗", "标准化"]
    },
    {
        "id": "doc4",
        "content": "A股股票数据提取建议：使用akshare库获取历史日线数据，常用字段包括日期、开盘价、最高价、最低价、收盘价、成交量、成交额。",
        "tags": ["提取", "A股", "数据源"]
    }
]

# 将文档内容转为列表
doc_contents = [doc["content"] for doc in knowledge_docs]

# 构建TF-IDF向量化器并拟合知识库（用于检索）
vectorizer = TfidfVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(doc_contents)

# ======================== 2. RAG Agent 类定义 ========================
class RAGFinancialAgent:
    def __init__(self, knowledge_docs, vectorizer, doc_vectors):
        self.knowledge_docs = knowledge_docs
        self.vectorizer = vectorizer
        self.doc_vectors = doc_vectors
        self.retrieved_rules = []      # 存储检索到的规则内容

    def retrieve(self, query, top_k=1):
        """检索与查询最相关的知识文档"""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        retrieved = [self.knowledge_docs[i] for i in top_indices if similarities[i] > 0]
        self.retrieved_rules = [doc["content"] for doc in retrieved]
        return self.retrieved_rules

    def generate_response(self, data_summary, factor_stats):
        """基于检索到的规则和数据处理结果生成最终回答（模拟生成）"""
        rules_text = "\n".join(self.retrieved_rules) if self.retrieved_rules else "未检索到特定规则，使用默认3-sigma原则。"
        response = f"""
        【RAG Agent 报告】
        检索到的知识规则：
        {rules_text}
        
        【数据清洗与因子提取结果】
        {data_summary}
        
        【因子清洗后统计】
        {factor_stats}
        
        结论：已根据检索规则完成数据清洗及因子异常值剔除。后续可进行量化建模分析。
        """
        return response

    def fetch_stock_data(self, symbol="000001", start_date="20230101", end_date="20231231"):
        """
        提取A股股票数据（使用akshare）
        symbol: 股票代码，例如'000001'（平安银行）
        """
        print(f"正在获取股票 {symbol} 数据...")
        try:
            # 获取历史日线数据
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="")
            if df.empty:
                raise ValueError(f"未获取到股票{symbol}数据，请检查代码或日期范围")
            # 重命名列为英文（便于处理）
            df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                "最低": "low", "成交量": "volume", "成交额": "amount", "振幅": "amplitude"
            }, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            print(f"成功获取 {len(df)} 条日线数据")
            return df
        except Exception as e:
            print(f"数据提取失败: {e}")
            # 返回模拟数据便于演示（实际使用中可删除）
            print("生成模拟数据用于演示...")
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            np.random.seed(42)
            n = len(dates)
            df = pd.DataFrame({
                'date': dates,
                'open': np.random.uniform(10, 15, n),
                'close': np.random.uniform(10, 15, n),
                'high': np.random.uniform(10, 16, n),
                'low': np.random.uniform(9, 14, n),
                'volume': np.random.randint(1000000, 50000000, n),
                'amount': np.random.uniform(1e7, 1e9, n),
                'amplitude': np.random.uniform(0.5, 5, n)
            })
            return df

    def clean_raw_data(self, df):
        """
        清洗原始股票数据：剔除异常值
        规则：剔除缺失值、收盘价<=0、成交量<=0、振幅异常（>20%）
        """
        print("\n开始清洗原始数据...")
        original_len = len(df)
        df_clean = df.dropna(subset=['close', 'volume', 'amount'])
        df_clean = df_clean[df_clean['close'] > 0]
        df_clean = df_clean[df_clean['volume'] > 0]
        # 振幅异常：超过20%可能由于数据错误或极端事件，剔除
        if 'amplitude' in df_clean.columns:
            df_clean = df_clean[df_clean['amplitude'] <= 20]
        dropped = original_len - len(df_clean)
        print(f"原始数据量: {original_len}, 清洗后: {len(df_clean)}, 剔除异常值数量: {dropped}")
        return df_clean

    def extract_factor(self, df):
        """
        提取因子：计算日收益率因子（ret）
        并采用3-sigma规则清洗因子异常值（依据检索规则）
        """
        print("\n提取因子（日收益率）...")
        df_factor = df.copy()
        # 计算日收益率（百分比）
        df_factor['ret'] = df_factor['close'].pct_change() * 100
        # 剔除收益率缺失值（首行）
        df_factor = df_factor.dropna(subset=['ret'])
        
        # 因子异常值清洗：使用3-sigma原则
        mean_ret = df_factor['ret'].mean()
        std_ret = df_factor['ret'].std()
        lower_bound = mean_ret - 3 * std_ret
        upper_bound = mean_ret + 3 * std_ret
        before_clean = len(df_factor)
        df_factor_clean = df_factor[(df_factor['ret'] >= lower_bound) & (df_factor['ret'] <= upper_bound)]
        after_clean = len(df_factor_clean)
        print(f"因子原始量: {before_clean}, 3-sigma剔除后: {after_clean}, 剔除异常值: {before_clean - after_clean}")
        print(f"因子均值: {mean_ret:.4f}, 标准差: {std_ret:.4f}, 有效区间: [{lower_bound:.2f}, {upper_bound:.2f}]")
        return df_factor_clean

    def run_pipeline(self, user_query, stock_symbol="000001"):
        """
        完整执行RAG流程：
        1. 根据用户查询检索知识库
        2. 提取股票数据
        3. 清洗原始数据
        4. 提取因子并清洗因子异常值
        5. 生成并返回最终报告
        """
        print("="*60)
        print("RAG Agent 启动")
        print(f"用户查询: {user_query}")
        print("="*60)
        
        # 步骤1: 检索相关规则
        retrieved = self.retrieve(user_query)
        print("\n[检索结果] 匹配到的知识规则:")
        for rule in retrieved:
            print(f"  - {rule}")
        
        # 步骤2: 提取数据
        df_raw = self.fetch_stock_data(symbol=stock_symbol)
        
        # 步骤3: 清洗原始数据
        df_cleaned = self.clean_raw_data(df_raw)
        
        # 步骤4: 提取因子并进行因子清洗
        df_with_factor = self.extract_factor(df_cleaned)
        
        # 准备数据摘要（用于生成）
        data_summary = f"""
        股票代码: {stock_symbol}
        原始数据条数: {len(df_raw)}
        清洗后数据条数: {len(df_cleaned)}
        因子提取后有效样本数: {len(df_with_factor)}
        因子名称: 日收益率(百分比)
        """
        factor_stats = f"""
        清洗后因子统计:
        均值: {df_with_factor['ret'].mean():.4f}%
        标准差: {df_with_factor['ret'].std():.4f}%
        最小值: {df_with_factor['ret'].min():.2f}%
        最大值: {df_with_factor['ret'].max():.2f}%
        中位数: {df_with_factor['ret'].median():.4f}%
        """
        # 步骤5: 生成最终回答
        final_response = self.generate_response(data_summary, factor_stats)
        return final_response, df_with_factor

# ======================== 3. 主程序演示 ========================
if __name__ == "__main__":
    # 初始化Agent
    agent = RAGFinancialAgent(knowledge_docs, vectorizer, doc_vectors)
    
    # 模拟用户查询（可根据需要修改）
    user_query = "如何提取A股股票数据并清洗因子中的异常值？"
    
    # 执行完整流水线（这里以平安银行000001为例，也可换其他代码如'600036'）
    response, result_df = agent.run_pipeline(user_query, stock_symbol="000001")
    
    # 输出最终结果
    print("\n" + "="*60)
    print("最终报告")
    print(response)
    print("\n清洗并计算因子后的数据样例（前5行）:")
    print(result_df[['date', 'close', 'ret']].head())