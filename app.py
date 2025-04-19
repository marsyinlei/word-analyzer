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
    """基于发音规则和字母组合的音节拆分"""
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
            'phonemes': [['h', 'ʌ', 'n'], ['t', 'ər']]
        },
        'actually': {
            'syllables': ['ac', 'tu', 'ally'],
            'phonemes': [['æ', 'k'], ['tʃ', 'u'], ['ə', 'l', 'i']]
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
        },
        'designer': {
            'syllables': ['de', 'sign', 'er'],
            'phonemes': [['d', 'i'], ['z', 'ˈaɪ', 'n'], ['ər']]
        },
        'heritage': {
            'syllables': ['he', 'ri', 'tage'],
            'phonemes': [['h', 'ˈɛ'], ['r', 'ɪ'], ['t', 'ɪ', 'dʒ']]
        }
    }
    
    # 首先检查是否是特殊单词
    if word in special_words:
        logger.info(f"特殊单词处理: {word}")
        return special_words[word]['phonemes'], special_words[word]['syllables']
    
    # 获取CMU原始发音
    if word not in d:
        return [phonemes], [word]  # 如果在CMU词典中找不到，返回整个单词
    
    original_cmu_phonemes = d[word][0]
    
    # 辅音组合和发音规则
    consonant_blends = {
        'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'sc', 'sk', 'sl', 
        'sm', 'sn', 'sp', 'st', 'sw', 'tr', 'tw', 'scr', 'spr', 'str', 'thr', 'shr', 'spl', 
        'squ', 'ch', 'sh', 'th', 'ph', 'wh', 'gn', 'kn', 'wr', 'qu', 'ck'
    }
    
    # 常见前缀
    common_prefixes = ['re', 'de', 'in', 'un', 'im', 'dis', 'mis', 'pre', 'ex', 'sub', 'inter', 
                       'super', 'over', 'under', 'non', 'anti', 'en', 'em', 'for', 'fore', 'pro', 
                       'co', 'con', 'com']
    
    # 常见后缀
    common_suffixes = ['tion', 'sion', 'ment', 'ness', 'ful', 'less', 'able', 'ible', 'ity', 'ty', 
                       'ly', 'ing', 'ed', 'er', 'or', 'ar', 'ist', 'ism', 'ate', 'al', 'ial', 'ic', 
                       'ical', 'ious', 'ous', 'ive', 'en', 'hood', 'ship', 'age']
    
    # 1. 首先提取元音位置
    vowels = 'aeiouy'
    vowel_positions = [i for i, char in enumerate(word) if char.lower() in vowels]
    
    # 如果没有元音，则整个单词作为一个音节
    if not vowel_positions:
        return [phonemes], [word]
    
    # 2. 识别前缀
    prefix = None
    prefix_end = 0
    for p in sorted(common_prefixes, key=len, reverse=True):
        if word.startswith(p) and len(p) < len(word) // 2:
            # 确保前缀的末尾为辅音或前缀后的字符为辅音
            if (p[-1] not in vowels) or (len(p) < len(word) and word[len(p)] not in vowels):
                prefix = p
                prefix_end = len(p)
                break
    
    # 3. 识别后缀
    suffix = None
    suffix_start = len(word)
    for s in sorted(common_suffixes, key=len, reverse=True):
        if word.endswith(s) and len(s) < len(word) // 2:
            suffix = s
            suffix_start = len(word) - len(s)
            break
    
    # 4. 处理中间部分
    middle = word[prefix_end:suffix_start]
    
    # 如果中间部分较短，将其作为一个整体
    if len(middle) <= 3:
        syllables = []
        if prefix:
            syllables.append(prefix)
        if middle:
            syllables.append(middle)
        if suffix:
            syllables.append(suffix)
        
        # 如果只有一个音节，则整个单词作为音节
        if not syllables:
            syllables = [word]
    else:
        # 5. 为中间部分拆分音节
        middle_syllables = []
        
        # 标识元音在CMU音标中的位置
        vowel_indices = []
        for i, phoneme in enumerate(original_cmu_phonemes):
            if any(phoneme.startswith(v) for v in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']):
                vowel_indices.append(i)
        
        # 根据元音位置和辅音规则确定音节边界
        middle_boundaries = []
        
        # 在中间部分查找字母元音位置
        middle_vowel_positions = [i - prefix_end for i in vowel_positions if prefix_end <= i < suffix_start]
        
        if len(middle_vowel_positions) <= 1:
            # 如果中间部分只有0-1个元音，将其作为一个整体
            middle_syllables = [middle]
        else:
            # 根据元音之间的辅音数量确定边界
            prev_end = 0
            for i in range(len(middle_vowel_positions) - 1):
                curr_pos = middle_vowel_positions[i]
                next_pos = middle_vowel_positions[i + 1]
                
                # 找到当前元音之后的第一个辅音
                cons_start = curr_pos
                while cons_start + 1 < len(middle) and cons_start + 1 < next_pos and middle[cons_start + 1] in vowels:
                    cons_start += 1
                
                # 当前元音到下一个元音之间的辅音数量
                cons_count = next_pos - cons_start - 1
                
                # 根据辅音数量确定音节边界
                if cons_count == 0:
                    # 如果没有辅音，在第一个元音后分界
                    boundary = curr_pos + 1
                elif cons_count == 1:
                    # 如果有一个辅音，在元音后的辅音后分界 (V-CV)
                    boundary = curr_pos + 2
                else:
                    # 检查辅音组合
                    consonant_pair = middle[cons_start+1:next_pos]
                    if consonant_pair in consonant_blends or any(blend in consonant_pair for blend in consonant_blends):
                        # 辅音组合保持在一起，边界在组合前
                        boundary = cons_start + 1
                    else:
                        # 在辅音中间分界
                        boundary = cons_start + 1 + cons_count // 2
                
                # 添加边界（相对于middle的起始位置）
                middle_syllables.append(middle[prev_end:boundary])
                prev_end = boundary
            
            # 添加最后一个音节
            if prev_end < len(middle):
                middle_syllables.append(middle[prev_end:])
        
        # 组合前缀、中间音节和后缀
        syllables = []
        if prefix:
            syllables.append(prefix)
        syllables.extend(middle_syllables)
        if suffix:
            syllables.append(suffix)
    
    # 6. 修正音节边界，确保每个音节至少有一个元音
    final_syllables = []
    for i, syl in enumerate(syllables):
        if not any(char in vowels for char in syl) and i > 0 and i < len(syllables) - 1:
            # 如果音节没有元音，尝试与前后音节合并
            if len(syllables[i-1]) <= len(syllables[i+1]):
                # 与前一个音节合并
                syllables[i-1] += syl
            else:
                # 与后一个音节合并
                syllables[i+1] = syl + syllables[i+1]
        else:
            final_syllables.append(syl)
    
    # 合并过短的音节（长度为1且不是元音）
    i = 1
    while i < len(final_syllables):
        if len(final_syllables[i]) == 1 and final_syllables[i] not in vowels:
            if i > 0:
                # 与前一个音节合并
                final_syllables[i-1] += final_syllables[i]
                final_syllables.pop(i)
            else:
                # 与后一个音节合并
                final_syllables[i+1] = final_syllables[i] + final_syllables[i+1]
                final_syllables.pop(i)
        else:
            i += 1
    
    # 7. 确认最终的音节集不会遗漏或重复字母
    joined = ''.join(final_syllables)
    if joined != word:
        # 如果有问题，回退到简单拆分
        final_syllables = []
        for i in range(len(vowel_positions)):
            if i == 0:
                # 第一个元音及其前面的所有辅音
                if vowel_positions[i] > 0:
                    final_syllables.append(word[:vowel_positions[i]+1])
                else:
                    final_syllables.append(word[0])
            elif i == len(vowel_positions) - 1:
                # 最后一个元音及其间隔和后面的所有字母
                start = vowel_positions[i-1] + 1
                final_syllables.append(word[start:])
            else:
                # 中间元音，从上一个元音后到当前元音
                start = vowel_positions[i-1] + 1
                end = vowel_positions[i] + 1
                final_syllables.append(word[start:end])
    
    # 8. 映射音标到音节
    # 获取原始CMU音标对应的IPA音标
    cmu_phonemes = original_cmu_phonemes
    
    # 将CMU音标映射到IPA音标
    cmu_to_ipa_map = {}
    for i, phoneme in enumerate(cmu_phonemes):
        if i < len(phonemes):
            cmu_to_ipa_map[phoneme] = phonemes[i]
    
    # 确保音节数量与元音数量匹配
    vowel_count = sum(1 for char in word if char in vowels)
    while len(final_syllables) > vowel_count and len(final_syllables) > 1:
        # 合并最短的相邻音节
        min_len = float('inf')
        min_idx = -1
        for i in range(len(final_syllables) - 1):
            pair_len = len(final_syllables[i]) + len(final_syllables[i+1])
            if pair_len < min_len:
                min_len = pair_len
                min_idx = i
        
        if min_idx >= 0:
            # 合并最短的相邻音节
            final_syllables[min_idx] += final_syllables[min_idx+1]
            final_syllables.pop(min_idx+1)
    
    # 为音节分配音标
    syllable_phonemes = []
    
    if len(final_syllables) == 1:
        # 单音节单词，所有音标分配给整个单词
        syllable_phonemes = [phonemes]
    else:
        # 尝试基于音素边界分配音标
        # 简单方案：根据音节长度比例分配音标
        total_length = sum(len(s) for s in final_syllables)
        phoneme_distribution = []
        
        current_idx = 0
        for syllable in final_syllables:
            # 按比例分配音标
            syllable_ratio = len(syllable) / total_length
            phoneme_count = max(1, round(syllable_ratio * len(phonemes)))
            
            # 确保不超出范围
            phoneme_count = min(phoneme_count, len(phonemes) - current_idx)
            
            syllable_phonemes.append(phonemes[current_idx:current_idx + phoneme_count])
            current_idx += phoneme_count
        
        # 如果还有剩余音标，添加到最后一个音节
        if current_idx < len(phonemes):
            syllable_phonemes[-1].extend(phonemes[current_idx:])
    
    logger.info(f"音节拆分（优化版）: {word} -> {final_syllables}")
    logger.info(f"音节及音标: {list(zip(final_syllables, syllable_phonemes))}")
    
    return syllable_phonemes, final_syllables

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