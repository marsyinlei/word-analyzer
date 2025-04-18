#!/bin/bash

# 获取当前目录的绝对路径
CURRENT_DIR=$(pwd)

# 确保在正确的目录
if [ ! -f "app.py" ]; then
    echo "错误：请在word_analyzer目录下运行此脚本"
    exit 1
fi

# 创建临时目录
TEMP_DIR=$(mktemp -d)
echo "创建临时目录: $TEMP_DIR"

# 复制所有必要文件到临时目录
echo "复制文件..."
cp -r app.py requirements.txt start.sh templates README.md $TEMP_DIR/

# 创建压缩包
echo "创建压缩包..."
cd $TEMP_DIR
zip -r "$CURRENT_DIR/word_analyzer.zip" *

# 清理临时目录
echo "清理临时文件..."
cd "$CURRENT_DIR"
rm -rf $TEMP_DIR

echo "打包完成！压缩包已创建：word_analyzer.zip" 