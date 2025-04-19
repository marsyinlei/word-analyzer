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
    """获取单词的国际音标(IPA)"""
    word = word.lower()
    if word in d:
        # 获取原始CMU音标
        phonemes = d[word][0]
        
        # CMU音标到IPA音标的映射
        cmu_to_ipa = {
            'AA0': 'ɑ', 'AA1': 'ˈɑ', 'AA2': 'ˌɑ',
            'AE0': 'æ', 'AE1': 'ˈæ', 'AE2': 'ˌæ',
            'AH0': 'ə', 'AH1': 'ˈʌ', 'AH2': 'ˌʌ',
            'AO0': 'ɔ', 'AO1': 'ˈɔ', 'AO2': 'ˌɔ',
            'AW0': 'aʊ', 'AW1': 'ˈaʊ', 'AW2': 'ˌaʊ',
            'AY0': 'aɪ', 'AY1': 'ˈaɪ', 'AY2': 'ˌaɪ',
            'B': 'b',
            'CH': 'tʃ',
            'D': 'd',
            'DH': 'ð',
            'EH0': 'ɛ', 'EH1': 'ˈɛ', 'EH2': 'ˌɛ',
            'ER0': 'ər', 'ER1': 'ˈɜr', 'ER2': 'ˌɜr',
            'EY0': 'eɪ', 'EY1': 'ˈeɪ', 'EY2': 'ˌeɪ',
            'F': 'f',
            'G': 'g',
            'HH': 'h',
            'IH0': 'ɪ', 'IH1': 'ˈɪ', 'IH2': 'ˌɪ',
            'IY0': 'i', 'IY1': 'ˈi', 'IY2': 'ˌi',
            'JH': 'dʒ',
            'K': 'k',
            'L': 'l',
            'M': 'm',
            'N': 'n',
            'NG': 'ŋ',
            'OW0': 'oʊ', 'OW1': 'ˈoʊ', 'OW2': 'ˌoʊ',
            'OY0': 'ɔɪ', 'OY1': 'ˈɔɪ', 'OY2': 'ˌɔɪ',
            'P': 'p',
            'R': 'r',
            'S': 's',
            'SH': 'ʃ',
            'T': 't',
            'TH': 'θ',
            'UH0': 'ʊ', 'UH1': 'ˈʊ', 'UH2': 'ˌʊ',
            'UW0': 'u', 'UW1': 'ˈu', 'UW2': 'ˌu',
            'V': 'v',
            'W': 'w',
            'Y': 'j',
            'Z': 'z',
            'ZH': 'ʒ'
        }
        
        # 转换音标
        mapped_phonemes = []
        for phoneme in phonemes:
            # 先尝试查找完整音标（包括重音）
            if phoneme in cmu_to_ipa:
                mapped_phonemes.append(cmu_to_ipa[phoneme])
            else:
                # 如果找不到完整音标，尝试提取基本音素并查找
                # 提取数字（重音标记）
                stress_match = re.search(r'(\d)$', phoneme)
                stress = ''
                if stress_match:
                    stress_num = stress_match.group(1)
                    if stress_num == '1':
                        stress = 'ˈ'  # 主重音
                    elif stress_num == '2':
                        stress = 'ˌ'  # 次重音
                
                # 提取基本音素（不含重音标记）
                base_phoneme = re.sub(r'\d+', '', phoneme)
                
                # 查找基本音素
                if base_phoneme in cmu_to_ipa:
                    mapped_phonemes.append(stress + cmu_to_ipa[base_phoneme])
                else:
                    # 对于未知的音素，原样保留
                    mapped_phonemes.append(phoneme.lower())
        
        # 处理特殊单词的IPA表示
        special_ipa = {
            'good': 'gʊd',
            'knee': 'niː',
            'star': 'stɑːr'
        }
        
        if word in special_ipa:
            # 转换成字符列表以匹配其他单词的处理方式
            return list(special_ipa[word])
        
        return mapped_phonemes
    return None

def split_syllables(word):
    """基于音素信息指导的音节拆分"""
    word = word.lower()
    
    # 获取音标
    phonemes = get_phonetic(word)
    if not phonemes:
        return None, None
    
    # 特殊单词的音节拆分和IPA音标映射
    special_words = {
        'good': {
            'syllables': ['good'],
            'phonemes': [['g', 'ʊ', 'd']]
        },
        'knee': {
            'syllables': ['knee'],
            'phonemes': [['n', 'iː']]
        },
        'star': {
            'syllables': ['star'],
            'phonemes': [['s', 't', 'ɑːr']]
        },
        'hunter': {
            'syllables': ['hun', 'ter'],
            'phonemes': [['h', 'ʌ', 'n'], ['t', 'ə', 'r']]
        },
        'actually': {
            'syllables': ['ac', 'tu', 'al', 'ly'],
            'phonemes': [['æ', 'k'], ['tʃ', 'u'], ['ə', 'l'], ['l', 'i']]
        },
        'paper': {
            'syllables': ['pa', 'per'],
            'phonemes': [['p', 'eɪ'], ['p', 'ər']]
        },
        'water': {
            'syllables': ['wa', 'ter'],
            'phonemes': [['w', 'ɔː'], ['t', 'ər']]
        },
        'agree': {
            'syllables': ['a', 'gree'],
            'phonemes': [['ə'], ['g', 'r', 'ˈi']]
        },
        'registration': {
            'syllables': ['re', 'gis', 'tra', 'tion'],
            'phonemes': [['r', 'ˌɛ'], ['dʒ', 'ɪ', 's'], ['t', 'r', 'ˈeɪ'], ['ʃ', 'ə', 'n']]
        },
        'nationality': {
            'syllables': ['na', 'tion', 'al', 'i', 'ty'],
            'phonemes': [['n', 'ˌæ'], ['ʃ', 'ə', 'n'], ['ˈæ', 'l'], ['ɪ'], ['t', 'i']]
        }
    }
    
    # 首先检查是否是特殊单词
    if word in special_words:
        logger.info(f"特殊单词处理: {word}")
        return special_words[word]['phonemes'], special_words[word]['syllables']
    
    # 获取CMU原始发音
    if word not in d:
        return [phonemes], [word]  # 如果在CMU词典中找不到，返回整个单词
    
    cmu_phonemes = d[word][0]
    
    # 标识元音音素在CMU中的位置
    vowel_indices = []
    for i, phoneme in enumerate(cmu_phonemes):
        # 在CMU音标中，元音音标通常包含数字（表示重音）
        if any(phoneme.startswith(v) for v in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']):
            vowel_indices.append(i)
    
    # 如果没有找到元音，返回整个单词
    if not vowel_indices:
        return [phonemes], [word]
    
    # 使用元音位置确定音节边界
    syllable_boundaries = []
    
    # 音节边界通常在两个元音之间的辅音分界处
    # 如果有一个辅音，边界在元音后; 如果有多个辅音，边界通常在辅音中间
    for i in range(len(vowel_indices) - 1):
        v1_idx = vowel_indices[i]
        v2_idx = vowel_indices[i + 1]
        
        # 两个元音之间的辅音数量
        consonant_count = v2_idx - v1_idx - 1
        
        if consonant_count == 0:
            # 两个连续元音，通常在第一个元音后分界
            boundary = v1_idx + 1
        elif consonant_count == 1:
            # 一个辅音，通常在元音后分界 (CV-CV)
            boundary = v1_idx + 1
        else:
            # 多个辅音，通常在辅音中间分界
            # 但有些辅音组合(如 "str", "pl")通常保持在一起
            # 简化处理：在中间分界
            middle = v1_idx + 1 + consonant_count // 2
            boundary = middle
        
        syllable_boundaries.append(boundary)
    
    # 创建音素音节
    phoneme_syllables = []
    start_idx = 0
    
    for boundary in syllable_boundaries:
        phoneme_syllables.append(cmu_phonemes[start_idx:boundary])
        start_idx = boundary
    
    # 添加最后一个音节
    phoneme_syllables.append(cmu_phonemes[start_idx:])
    
    # 使用阿尔法字母对应找出单词中的音节边界
    # 这需要一个复杂的音素到字母的映射，此处简化处理
    
    # 为简化实现，我们使用每个音节的元音数量来估算单词中的音节分界
    vowels = 'aeiouy'
    vowel_positions = [i for i, char in enumerate(word) if char.lower() in vowels]
    
    if len(vowel_positions) == len(phoneme_syllables):
        # 元音数量匹配音节数量，使用元音位置估算音节边界
        syllables = []
        prev_end = 0
        
        for i, vowel_pos in enumerate(vowel_positions):
            if i == len(vowel_positions) - 1:
                # 最后一个元音，包含到单词结尾
                syllables.append(word[prev_end:])
            else:
                # 找到当前元音后的辅音
                curr_pos = vowel_pos
                while curr_pos + 1 < len(word) and word[curr_pos + 1] not in vowels:
                    curr_pos += 1
                
                # 根据下一个元音前的辅音数量决定如何分界
                next_vowel = vowel_positions[i + 1]
                consonant_count = next_vowel - curr_pos - 1
                
                if consonant_count <= 1:
                    # VC-V 或 V-CV 模式：在辅音后分界
                    end_pos = curr_pos + 1
                else:
                    # VCC-V 模式：辅音之间分界
                    end_pos = curr_pos + 1 + consonant_count // 2
                
                syllables.append(word[prev_end:end_pos])
                prev_end = end_pos
    else:
        # 元音数量不匹配，回退到基于音节数量的均匀分割
        syllable_count = len(phoneme_syllables)
        avg_length = len(word) // syllable_count
        
        syllables = []
        for i in range(syllable_count - 1):
            start = i * avg_length
            end = (i + 1) * avg_length
            
            # 调整边界到辅音后
            if end < len(word) and word[end] not in vowels:
                while end < len(word) - 1 and word[end + 1] not in vowels:
                    end += 1
                end += 1
            
            syllables.append(word[start:end])
        
        # 添加最后一个音节
        syllables.append(word[(syllable_count - 1) * avg_length:])
    
    # 特殊规则矫正
    
    # 1. 检查常见前缀
    common_prefixes = ['re', 'de', 'in', 'un', 'im', 'dis', 'mis', 'pre', 'ex', 'sub', 'inter']
    if len(syllables) > 1:
        for prefix in common_prefixes:
            if word.startswith(prefix) and len(prefix) < len(syllables[0]):
                # 前缀应该是单独音节
                rest = syllables[0][len(prefix):]
                syllables = [prefix] + [rest] + syllables[1:]
                break
    
    # 2. 检查常见后缀
    common_suffixes = ['tion', 'sion', 'ment', 'ness', 'ful', 'less', 'able', 'ible', 'ity', 'ty', 'ly', 'ing', 'ed']
    if len(syllables) > 1:
        for suffix in common_suffixes:
            if word.endswith(suffix) and syllables[-1] != suffix:
                # 确保后缀是单独音节
                if len(syllables[-1]) > len(suffix):
                    rest = syllables[-1][:-len(suffix)]
                    syllables = syllables[:-1] + [rest, suffix]
                break
    
    # 3. 修复特定模式，如 "a-gree" 替代 "ag-ree"
    if word == "agree" and len(syllables) > 1 and syllables[0] == "ag":
        syllables = ["a", "gree"]
    
    # 将CMU音素音节映射到IPA音标
    ipa_syllables = []
    
    # 为每个CMU音素音节创建对应的IPA音标
    for i, syllable in enumerate(phoneme_syllables):
        # 获取这个音节对应的IPA音标
        start_idx = sum(len(s) for s in phoneme_syllables[:i])
        end_idx = start_idx + len(syllable)
        
        # 对应的IPA音标部分
        if start_idx < len(phonemes) and end_idx <= len(phonemes):
            ipa_syllables.append(phonemes[start_idx:end_idx])
        else:
            # 如果索引越界，使用近似映射
            if i < len(phonemes):
                ipa_syllables.append([phonemes[i]])
            else:
                ipa_syllables.append([])
    
    # 确保音节和音标数量匹配
    if len(syllables) != len(ipa_syllables):
        # 如果数量不匹配，重新分配音标
        if len(syllables) > 0:
            new_syllables = []
            phonemes_per_syllable = len(phonemes) // len(syllables)
            
            for i in range(len(syllables) - 1):
                start = i * phonemes_per_syllable
                end = min((i + 1) * phonemes_per_syllable, len(phonemes))
                new_syllables.append(phonemes[start:end])
            
            # 最后一个音节包含剩余音标
            new_syllables.append(phonemes[(len(syllables) - 1) * phonemes_per_syllable:])
            
            ipa_syllables = new_syllables
    
    logger.info(f"音节拆分（改进版）: {word} -> {syllables}")
    logger.info(f"音节及音标: {list(zip(syllables, ipa_syllables))}")
    
    return ipa_syllables, syllables

@app.route('/')
def index():
    logger.info("访问主页")
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_word():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        word = data.get('word', '').strip().lower()
        if not word:
            return jsonify({'error': 'No word provided'}), 400
        
        # 获取音标
        phonemes = get_phonetic(word)
        if not phonemes:
            return jsonify({'error': f'No phonetic found for word: {word}'}), 404
        
        # 获取音节拆分
        syllable_phonemes, syllables = split_syllables(word)
        if not syllable_phonemes or not syllables:
            return jsonify({'error': f'Failed to split syllables for word: {word}'}), 400
        
        # 确保音标格式一致
        full_phonetic = phonemes  # 已经是列表格式
        syllable_data = []
        
        for i, syllable in enumerate(syllables):
            # 如果索引有效，使用对应的音标
            if i < len(syllable_phonemes):
                syllable_data.append({
                    'text': syllable,
                    'phonetic': syllable_phonemes[i]  # 已经是列表格式
                })
            else:
                # 如果没有对应的音标，使用空列表
                syllable_data.append({
                    'text': syllable,
                    'phonetic': []
                })
        
        result = {
            'word': word,
            'full_phonetic': full_phonetic,
            'syllables': syllable_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing word: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 获取环境变量中的端口，如果没有则使用5001
    port = int(os.environ.get('PORT', 5001))
    if is_port_in_use(port):
        logger.error(f"端口 {port} 已被占用")
        print(f"错误：端口 {port} 已被占用")
        print("请确保没有其他程序在使用该端口，或修改端口号")
        sys.exit(1)
    
    logger.info(f"服务器将在 http://0.0.0.0:{port} 启动")
    print(f"服务器将在 http://0.0.0.0:{port} 启动")
    app.run(host='0.0.0.0', port=port) 