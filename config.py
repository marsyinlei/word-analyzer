# API配置
API_CONFIG = {
    # 服务器配置
    "host": "127.0.0.1",  # 服务器地址
    "port": 5000,         # 服务器端口
    
    # API端点配置
    "endpoints": {
        "analyze": "/analyze"  # 分析单词的端点
    },
    
    # 请求头配置
    "headers": {
        "Content-Type": "application/json"
    }
} 