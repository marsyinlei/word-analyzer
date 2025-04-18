#!/bin/bash

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到Python3，请先安装Python3"
    exit 1
fi

# 检查pip是否安装
if ! command -v pip3 &> /dev/null; then
    echo "错误：未找到pip3，请先安装pip3"
    exit 1
fi

# 安装依赖
echo "正在安装依赖..."
pip3 install -r requirements.txt

# 下载NLTK数据
echo "正在下载NLTK数据..."
python3 -c "import nltk; nltk.download('cmudict')"

echo "安装完成！" 