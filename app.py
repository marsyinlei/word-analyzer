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
    """基于音标和英语单词结构拆分音节"""
    word = word.lower()
    # 获取音标
    phonemes = get_phonetic(word)
    if not phonemes:
        return None, None
    
    # 初始化常见的英语单词结构（前缀、后缀）映射到IPA音标
    common_suffixes = {
        'ture': ['tʃ', 'ər'],      # lecture, nature
        'tion': ['ʃ', 'ə', 'n'],   # nation, station
        'sion': ['ʒ', 'ə', 'n'],   # vision, fusion
        'cious': ['ʃ', 'ə', 's'],  # delicious
        'tious': ['ʃ', 'ə', 's'],  # cautious
        'cial': ['ʃ', 'ə', 'l'],   # social
        'tial': ['ʃ', 'ə', 'l'],   # partial
        'er': ['ər'],              # hunter, paper
        'or': ['ɔr'],              # doctor, actor
        'ar': ['ɑr'],              # grammar, solar
        'ly': ['l', 'i'],          # quickly, badly
        'ment': ['m', 'ə', 'n', 't'], # document, payment
        'ness': ['n', 'ə', 's'],   # kindness, darkness
        'ful': ['f', 'ʊ', 'l'],    # wonderful, beautiful
        'less': ['l', 'ə', 's'],   # useless, careless
        'able': ['ə', 'b', 'ə', 'l'], # comfortable, reliable
        'ible': ['ə', 'b', 'ə', 'l'], # possible, terrible
        'al': ['ə', 'l'],          # animal, musical
        'ial': ['i', 'ə', 'l'],    # commercial, material
        'ic': ['ɪ', 'k'],          # public, basic
        'ical': ['ɪ', 'k', 'ə', 'l'], # logical, musical
        'ive': ['ɪ', 'v'],         # active, native
        'ty': ['t', 'i'],          # nationality, ability
        'ity': ['ɪ', 't', 'i'],    # priority, quality
        'ary': ['ɛ', 'r', 'i'],    # dictionary, secretary
        'ery': ['ɛ', 'r', 'i'],    # brewery, bakery
        'ory': ['ɔ', 'r', 'i'],    # directory, factory
    }
    
    common_prefixes = {
        'ex': ['ɛ', 'k', 's'],     # exchange, example
        'pre': ['p', 'r', 'i'],    # prefix, predict
        'de': ['d', 'i'],          # decide, define
        're': ['r', 'i'],          # return, repeat
        'un': ['ʌ', 'n'],          # untie, undo
        'in': ['ɪ', 'n'],          # inside, input
        'im': ['ɪ', 'm'],          # impossible, immoral
        'il': ['ɪ', 'l'],          # illegal, illegible
        'ir': ['ɪ', 'r'],          # irregular, irresponsible
        'dis': ['d', 'ɪ', 's'],    # discover, disappear
        'mis': ['m', 'ɪ', 's'],    # mistake, misuse
        'non': ['n', 'ɑ', 'n'],    # nonstop, nonexistent
        'over': ['o', 'v', 'ər'],  # overcome, overlook
        'under': ['ʌ', 'n', 'd', 'ər'], # understand, underestimate
        'sub': ['s', 'ʌ', 'b'],    # submarine, submerge
        'super': ['s', 'u', 'p', 'ər'], # supermarket, superpower
        'inter': ['ɪ', 'n', 't', 'ər'], # international, interrupt
        'anti': ['æ', 'n', 't', 'i'], # antibiotic, antifreeze
    }
    
    # 这些是单音节元音前缀
    vowel_prefixes = ['a', 'e', 'i', 'o', 'u']
    
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
        }
    }
    
    # 首先检查是否是特殊单词
    if word in special_words:
        logger.info(f"特殊单词处理: {word}")
        return special_words[word]['phonemes'], special_words[word]['syllables']
    
    # 使用pyphen库进行音节拆分
    hyphenated = dic.inserted(word)
    syllables = hyphenated.split('-')
    
    # 如果pyphen无法正确拆分，回退到基本拆分
    if len(syllables) == 1 and len(word) > 3:
        # 尝试基于元音辅音模式进行基本拆分
        vowels = 'aeiouy'
        consonants = 'bcdfghjklmnpqrstvwxz'
        syllable_boundaries = []
        
        for i in range(1, len(word) - 1):
            # 如果当前字符是辅音，前一个是元音，后一个是元音，可能是音节边界
            if (word[i] in consonants and 
                word[i-1] in vowels and 
                word[i+1] in vowels):
                syllable_boundaries.append(i + 1)  # 在辅音后断开
            # 如果有两个连续辅音，可能在它们之间断开
            elif (word[i] in consonants and 
                  word[i-1] in consonants and 
                  i > 1 and word[i-2] in vowels and 
                  i < len(word) - 1 and word[i+1] in vowels):
                syllable_boundaries.append(i)
        
        # 根据边界拆分单词
        if syllable_boundaries:
            new_syllables = []
            start = 0
            for boundary in sorted(syllable_boundaries):
                if boundary > start:
                    new_syllables.append(word[start:boundary])
                    start = boundary
            if start < len(word):
                new_syllables.append(word[start:])
            syllables = new_syllables
    
    logger.info(f"拆分音节: {word} -> {syllables}")
    
    # 为每个音节分配音标
    syllable_phonemes = []
    
    # 如果没有成功拆分音节或只有一个音节，将所有音标分配给单个音节
    if not syllables or len(syllables) == 1:
        syllable_phonemes = [phonemes]
        return syllable_phonemes, [word]
    
    # 音标总长度
    total_phonemes = len(phonemes)
    # 平均每个音节的音标数量
    phonemes_per_syllable = max(1, total_phonemes // len(syllables))
    
    # 为每个音节分配音标
    for i in range(len(syllables) - 1):
        start = i * phonemes_per_syllable
        end = min((i + 1) * phonemes_per_syllable, total_phonemes)
        syllable_phonemes.append(phonemes[start:end])
    
    # 最后一个音节获取剩余所有音标
    start = (len(syllables) - 1) * phonemes_per_syllable
    syllable_phonemes.append(phonemes[start:])
    
    logger.info(f"音节及音标: {list(zip(syllables, syllable_phonemes))}")
    return syllable_phonemes, syllables

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