from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # 添加CORS支持
from nltk.corpus import cmudict
import nltk
import re
import socket
import sys
import logging
import pyphen
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 下载CMU发音词典
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    logger.info("正在下载CMU发音词典...")
    nltk.download('cmudict')

# 获取CMU发音词典
try:
    d = cmudict.dict()
    logger.info("成功加载CMU发音词典")
except Exception as e:
    logger.error(f"加载CMU发音词典失败: {str(e)}")
    sys.exit(1)

# 初始化pyphen
dic = pyphen.Pyphen(lang='en_US')

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def get_phonetic(word):
    """获取单词的音标"""
    word = word.lower()
    if word in d:
        return d[word][0]  # 返回第一个发音
    return None

def split_syllables(word):
    """按音节拆分单词"""
    word = word.lower()
    # 获取音标
    phonemes = get_phonetic(word)
    if not phonemes:
        return None, None
    
    # 使用pyphen进行音节拆分
    hyphenated = dic.inserted(word)
    syllables = hyphenated.split('-')
    
    # 打印调试信息
    logger.info(f"单词: {word}")
    logger.info(f"音节拆分: {syllables}")
    logger.info(f"音标: {phonemes}")
    
    # 定义元音音素
    vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    
    # 手动处理一些特殊情况的词典
    special_words = {
        'exchange': ([
            ['IH0', 'K', 'S'],  # ex
            ['CH', 'EY1', 'N', 'JH']  # change
        ], ['ex', 'change']),
        'word': ([
            ['W', 'ER1', 'D']  # word
        ], ['word']),
        'actually': ([
            ['AE1', 'K'],  # ac
            ['CH', 'AH0', 'W'],  # tu
            ['AH0'],  # a
            ['L', 'IY0']  # lly
        ], ['ac', 'tu', 'a', 'lly']),
        'beautiful': ([
            ['B', 'Y', 'UW1'],  # beau
            ['T', 'IH0'],  # ti
            ['F', 'AH0', 'L']  # ful
        ], ['beau', 'ti', 'ful'])
    }
    
    # 检查是否是特殊单词
    if word in special_words:
        return special_words[word]
    
    # 根据每个音节包含的字母数，将音标映射到音节
    syllable_phonemes = []
    
    # 统计元音数量
    vowel_count = sum(1 for p in phonemes if re.sub(r'\d+', '', p) in vowels)
    
    # 如果元音数量与音节数量相等，则可以使用元音位置来分割
    if vowel_count == len(syllables):
        # 找到所有元音音素的位置
        vowel_positions = []
        for i, phoneme in enumerate(phonemes):
            base = re.sub(r'\d+', '', phoneme)
            if base in vowels:
                vowel_positions.append(i)
        
        # 根据元音位置分配音标到音节
        start = 0
        for i in range(len(syllables)):
            if i == len(syllables) - 1:
                # 最后一个音节
                syllable_phonemes.append(phonemes[start:])
            else:
                # 找到下一个音节的开始位置
                next_vowel_pos = vowel_positions[i + 1]
                vowel_pos = vowel_positions[i]
                
                # 对于辅音，我们需要正确分配到前后音节
                # 一般规则：如果两个元音之间有两个辅音，通常第一个辅音属于前一个音节，第二个属于后一个音节
                if next_vowel_pos - vowel_pos > 2:
                    # 有多个辅音，找到合适的分割点
                    split_pos = vowel_pos + 1
                    while split_pos < next_vowel_pos - 1:
                        split_pos += 1
                else:
                    # 只有一个辅音，归入后一个音节
                    split_pos = vowel_pos + 1
                
                syllable_phonemes.append(phonemes[start:split_pos])
                start = split_pos
    else:
        # 元音数量与音节数量不匹配，尝试基于字母长度分配
        total_letters = len(word)
        total_phonemes = len(phonemes)
        
        start = 0
        for syllable in syllables:
            # 计算当前音节占总单词的比例
            syllable_ratio = len(syllable) / total_letters
            # 根据比例计算应分配的音素数量
            phoneme_count = max(1, round(syllable_ratio * total_phonemes))
            end = min(start + phoneme_count, total_phonemes)
            
            syllable_phonemes.append(phonemes[start:end])
            start = end
            
            # 如果已经分配完所有音素，跳出循环
            if start >= total_phonemes:
                break
    
    return syllable_phonemes, syllables

def natural_reading_split(word, phonemes):
    """对单词进行自然拼读拆分"""
    # 定义元音音素
    vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    
    # 定义辅音组合
    consonant_clusters = {
        'CH': 'ch',  # /tʃ/
        'SH': 'sh',  # /ʃ/
        'TH': 'th',  # /θ/
        'ZH': 'zh',  # /ʒ/
        'NG': 'ng',  # /ŋ/
        'PH': 'ph',  # /f/
        'GH': 'gh',  # 通常是静音或 /f/
        'CK': 'ck',  # /k/
        'WR': 'wr',  # /r/
        'KN': 'kn',  # /n/
        'GN': 'gn',  # /n/
        'MB': 'mb',  # /m/
        'SC': 'sc',  # /s/
        'TCH': 'tch',  # /tʃ/
    }
    
    # 特殊单词字典
    special_readings = {
        'exchange': [
            {'text': 'e', 'phonemes': ['IH0']},
            {'text': 'x', 'phonemes': ['K', 'S']},
            {'text': 'ch', 'phonemes': ['CH']},
            {'text': 'a', 'phonemes': ['EY1']},
            {'text': 'n', 'phonemes': ['N']},
            {'text': 'ge', 'phonemes': ['JH']}
        ],
        'word': [
            {'text': 'w', 'phonemes': ['W']},
            {'text': 'or', 'phonemes': ['ER1']},
            {'text': 'd', 'phonemes': ['D']}
        ],
        'actually': [
            {'text': 'a', 'phonemes': ['AE1']},
            {'text': 'c', 'phonemes': ['K']},
            {'text': 't', 'phonemes': ['T']},
            {'text': 'u', 'phonemes': ['CH', 'AH0']},
            {'text': 'a', 'phonemes': ['AH0']},
            {'text': 'll', 'phonemes': ['L']},
            {'text': 'y', 'phonemes': ['IY0']}
        ],
        'beautiful': [
            {'text': 'b', 'phonemes': ['B']},
            {'text': 'eau', 'phonemes': ['Y', 'UW1']},
            {'text': 't', 'phonemes': ['T']},
            {'text': 'i', 'phonemes': ['IH0']},
            {'text': 'f', 'phonemes': ['F']},
            {'text': 'u', 'phonemes': ['AH0']},
            {'text': 'l', 'phonemes': ['L']}
        ]
    }
    
    # 检查是否是特殊单词
    if word in special_readings:
        return special_readings[word]
    
    # 智能拆分算法
    reading_units = []
    
    # 将单词拆分为字母
    word_chars = list(word)
    
    # 初始化索引和当前单元
    i = 0
    phoneme_index = 0
    
    while i < len(word_chars):
        # 检查是否是辅音组合
        match_found = False
        for cluster, representation in sorted(consonant_clusters.items(), key=lambda x: len(x[0]), reverse=True):
            cluster_length = len(cluster)
            if i + cluster_length <= len(word_chars) and ''.join(word_chars[i:i+cluster_length]).lower() == cluster.lower():
                # 找到一个辅音组合
                if phoneme_index < len(phonemes):
                    # 分配对应的音素
                    phoneme_slice = [phonemes[phoneme_index]]
                    phoneme_index += 1
                else:
                    phoneme_slice = []
                
                reading_units.append({
                    'text': representation,
                    'phonemes': phoneme_slice
                })
                
                i += cluster_length
                match_found = True
                break
        
        if match_found:
            continue
        
        # 处理单个字母
        if i < len(word_chars):
            char = word_chars[i]
            
            # 检查是否是元音
            is_vowel = char.lower() in 'aeiou'
            
            if phoneme_index < len(phonemes):
                # 分配对应的音素
                current_phoneme = phonemes[phoneme_index]
                phoneme_base = re.sub(r'\d+', '', current_phoneme)
                
                if is_vowel and phoneme_base in vowels:
                    # 元音对应元音音素
                    reading_units.append({
                        'text': char,
                        'phonemes': [current_phoneme]
                    })
                    phoneme_index += 1
                else:
                    # 辅音或不匹配的情况
                    reading_units.append({
                        'text': char,
                        'phonemes': [current_phoneme] if phoneme_index < len(phonemes) else []
                    })
                    phoneme_index += 1
            else:
                # 已经没有更多的音素
                reading_units.append({
                    'text': char,
                    'phonemes': []
                })
            
            i += 1
    
    # 确保所有音素都被使用
    remaining_phonemes = phonemes[phoneme_index:]
    if remaining_phonemes and reading_units:
        # 将剩余音素分配给最后一个单元
        reading_units[-1]['phonemes'].extend(remaining_phonemes)
    
    return reading_units

def merge_adjacent_units(reading_units):
    """合并相邻的拼读单元，处理特殊组合"""
    # 定义需要合并的模式
    patterns = [
        {'chars': ['t', 'h'], 'output': 'th'},
        {'chars': ['s', 'h'], 'output': 'sh'},
        {'chars': ['c', 'h'], 'output': 'ch'},
        {'chars': ['p', 'h'], 'output': 'ph'},
        {'chars': ['w', 'h'], 'output': 'wh'},
        {'chars': ['c', 'k'], 'output': 'ck'},
        {'chars': ['e', 'a'], 'output': 'ea'},
        {'chars': ['a', 'i'], 'output': 'ai'},
        {'chars': ['e', 'e'], 'output': 'ee'},
        {'chars': ['o', 'o'], 'output': 'oo'},
        {'chars': ['o', 'u'], 'output': 'ou'},
        {'chars': ['a', 'u'], 'output': 'au'},
        {'chars': ['e', 'i'], 'output': 'ei'},
        {'chars': ['i', 'e'], 'output': 'ie'},
        {'chars': ['q', 'u'], 'output': 'qu'},
    ]
    
    if len(reading_units) < 2:
        return reading_units
    
    merged_units = [reading_units[0]]
    
    for i in range(1, len(reading_units)):
        current_unit = reading_units[i]
        prev_unit = merged_units[-1]
        
        # 检查是否匹配任何模式
        match_found = False
        for pattern in patterns:
            if len(prev_unit['text']) == 1 and len(current_unit['text']) == 1 and \
               prev_unit['text'].lower() == pattern['chars'][0] and current_unit['text'].lower() == pattern['chars'][1]:
                # 合并单元
                merged_units[-1] = {
                    'text': pattern['output'],
                    'phonemes': prev_unit['phonemes'] + current_unit['phonemes']
                }
                match_found = True
                break
        
        if not match_found:
            merged_units.append(current_unit)
    
    return merged_units

def phoneme_to_letter(phoneme):
    """将单个音素转换为字母表示"""
    phoneme_to_letter_map = {
        'AA': 'a', 'AE': 'a', 'AH': 'a', 'AO': 'o', 'AW': 'ow',
        'AY': 'ay', 'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'th',
        'EH': 'e', 'ER': 'er', 'EY': 'ey', 'F': 'f', 'G': 'g',
        'HH': 'h', 'IH': 'i', 'IY': 'ee', 'JH': 'j', 'K': 'k',
        'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ng', 'OW': 'o',
        'OY': 'oy', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'sh',
        'T': 't', 'TH': 'th', 'UH': 'u', 'UW': 'oo', 'V': 'v',
        'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh'
    }
    return phoneme_to_letter_map.get(phoneme, phoneme.lower())

def phonemes_to_ipa(phonemes):
    """将CMU音素转换为IPA音标"""
    # CMU到IPA的映射
    cmu_to_ipa = {
        'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aʊ',
        'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
        'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'g',
        'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
        'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
        'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ',
        'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
        'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
    }
    
    ipa = []
    for phoneme in phonemes:
        base = re.sub(r'\d+', '', phoneme)  # 移除重音标记
        if base in cmu_to_ipa:
            ipa.append(cmu_to_ipa[base])
    return ''.join(ipa)

def phonemes_to_letters(phonemes):
    """将音素转换为字母表示"""
    # 音素到字母的映射
    phoneme_to_letter = {
        'AA': 'a', 'AE': 'a', 'AH': 'a', 'AO': 'o', 'AW': 'ow',
        'AY': 'ay', 'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'th',
        'EH': 'e', 'ER': 'er', 'EY': 'ey', 'F': 'f', 'G': 'g',
        'HH': 'h', 'IH': 'i', 'IY': 'ee', 'JH': 'j', 'K': 'k',
        'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ng', 'OW': 'o',
        'OY': 'oy', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'sh',
        'T': 't', 'TH': 'th', 'UH': 'u', 'UW': 'oo', 'V': 'v',
        'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh'
    }
    
    # 特殊组合的映射
    special_combinations = {
        'TCH': 'tch',
        'DGE': 'dge',
        'CK': 'ck',
        'PH': 'ph',
        'GH': 'gh',
        'QU': 'qu',
        'WR': 'wr',
        'WH': 'wh',
        'X': 'x',
        'EX': 'ex'
    }
    
    letters = []
    i = 0
    while i < len(phonemes):
        if i < len(phonemes) - 1:
            current = re.sub(r'\d+', '', phonemes[i])
            next_phoneme = re.sub(r'\d+', '', phonemes[i+1])
            combination = current + next_phoneme
            
            # 检查特殊组合
            if combination in special_combinations:
                letters.append(special_combinations[combination])
                i += 2
                continue
        
        phoneme = re.sub(r'\d+', '', phonemes[i])
        if phoneme in phoneme_to_letter:
            letters.append(phoneme_to_letter[phoneme])
        i += 1
    
    return ''.join(letters)

@app.route('/')
def index():
    logger.info("访问主页")
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_word():
    try:
        data = request.get_json()
        if not data:
            logger.error("未收到JSON数据")
            return jsonify({'error': '无效的请求数据'}), 400
            
        word = data.get('word', '').strip()
        logger.info(f"分析单词: {word}")
        
        if not word:
            return jsonify({'error': '请输入单词'}), 400
        
        phonetic = get_phonetic(word)
        if not phonetic:
            logger.warning(f"找不到单词 '{word}' 的音标")
            return jsonify({'error': f'找不到单词 "{word}" 的音标'}), 404
        
        syllable_phonemes, syllable_texts = split_syllables(word)
        if not syllable_phonemes:
            logger.warning(f"无法拆分单词 '{word}' 的音节")
            return jsonify({'error': f'无法拆分单词 "{word}" 的音节'}), 400
        
        # 获取完整单词的IPA音标
        full_ipa = phonemes_to_ipa(phonetic)
        
        # 获取每个音节的IPA音标
        syllables = []
        for i, syllable_phoneme in enumerate(syllable_phonemes):
            syllable_ipa = phonemes_to_ipa(syllable_phoneme)
            syllable_text = syllable_texts[i] if i < len(syllable_texts) else phonemes_to_letters(syllable_phoneme)
            syllables.append({
                'text': syllable_text,
                'phonetic': syllable_ipa
            })
        
        # 获取自然拼读拆分
        natural_reading_units = natural_reading_split(word, phonetic)
        
        # 合并相邻的自然拼读单元，处理组合
        natural_reading_units = merge_adjacent_units(natural_reading_units)
        
        natural_reading = []
        for unit in natural_reading_units:
            unit_ipa = phonemes_to_ipa(unit['phonemes'])
            natural_reading.append({
                'text': unit['text'],
                'phonetic': unit_ipa
            })
        
        result = {
            'word': word,
            'full_phonetic': full_ipa,
            'syllables': syllables,
            'natural_reading': natural_reading
        }
        
        logger.info(f"成功分析单词 '{word}'")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

if __name__ == '__main__':
    # 获取环境变量中的端口，如果没有则使用10000（Render的默认端口）
    port = int(os.environ.get('PORT', 10000))
    if is_port_in_use(port):
        logger.error(f"端口 {port} 已被占用")
        print(f"错误：端口 {port} 已被占用")
        print("请确保没有其他程序在使用该端口，或修改端口号")
        sys.exit(1)
    
    logger.info(f"服务器将在 http://0.0.0.0:{port} 启动")
    print(f"服务器将在 http://0.0.0.0:{port} 启动")
    app.run(host='0.0.0.0', port=port) 