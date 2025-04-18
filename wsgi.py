import sys

# 添加项目目录到Python路径
path = '/home/your_username/word_analyzer'
if path not in sys.path:
    sys.path.append(path)

# 导入应用
from app import app as application 