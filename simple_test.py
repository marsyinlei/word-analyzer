import requests

# 发送测试请求
url = "http://127.0.0.1:5000/analyze"  # 使用127.0.0.1而不是localhost
data = {"word": "beautiful"}
headers = {"Content-Type": "application/json"}

try:
    print("正在发送请求...")
    response = requests.post(url, json=data, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
except Exception as e:
    print(f"发生错误: {e}") 