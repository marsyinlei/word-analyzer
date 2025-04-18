import requests
from config import API_CONFIG

class WordAnalyzerClient:
    def __init__(self, host=None, port=None):
        """初始化API客户端"""
        self.host = host or API_CONFIG["host"]
        self.port = port or API_CONFIG["port"]
        self.base_url = f"http://{self.host}:{self.port}"
        self.headers = API_CONFIG["headers"]
    
    def analyze_word(self, word):
        """分析单词
        
        Args:
            word (str): 要分析的单词
            
        Returns:
            dict: 分析结果，包含：
                - word: 原始单词
                - syllables: 音节拆分
                - natural_split: 自然拼读拆分
                - phonetic: 音标
        """
        url = f"{self.base_url}{API_CONFIG['endpoints']['analyze']}"
        data = {"word": word}
        
        try:
            response = requests.post(url, json=data, headers=self.headers)
            response.raise_for_status()  # 检查HTTP错误
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {str(e)}")
            return None
    
    def analyze_words(self, words):
        """批量分析单词
        
        Args:
            words (list): 要分析的单词列表
            
        Returns:
            list: 分析结果列表
        """
        results = []
        for word in words:
            result = self.analyze_word(word)
            if result:
                results.append(result)
        return results

# 使用示例
if __name__ == "__main__":
    # 创建客户端实例
    client = WordAnalyzerClient()
    
    # 分析单个单词
    result = client.analyze_word("example")
    print("单个单词分析结果:")
    print(result)
    
    # 批量分析单词
    words = ["beautiful", "computer", "programming"]
    results = client.analyze_words(words)
    print("\n批量分析结果:")
    for result in results:
        print(result) 