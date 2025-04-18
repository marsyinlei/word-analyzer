import requests
import json
import sys

def test_api():
    url = "http://localhost:5000/analyze"
    headers = {"Content-Type": "application/json"}
    
    try:
        # 测试用例1：基本单词
        print("正在测试API连接...")
        data = {"word": "example"}
        response = requests.post(url, headers=headers, json=data)
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            print("\n测试用例1 - example:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
        
    except requests.exceptions.ConnectionError:
        print("无法连接到服务器，请确保服务器正在运行")
        print("运行命令: python app.py")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    test_api() 