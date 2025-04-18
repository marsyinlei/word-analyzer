#!/bin/bash

# 检查是否在正确的目录
if [ ! -f "app.py" ]; then
    echo "错误：请在word_analyzer目录下运行此脚本"
    exit 1
fi

# 检查端口是否被占用
PORT=5001
if lsof -i :$PORT > /dev/null; then
    echo "警告：端口 $PORT 已被占用"
    echo "正在尝试关闭占用端口的进程..."
    lsof -ti :$PORT | xargs kill -9
    sleep 2
fi

# 启动Flask应用
echo "启动单词音标分析器..."
python3 app.py 