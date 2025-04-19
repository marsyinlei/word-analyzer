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
    """基于发音规则和人类阅读习惯的音节拆分"""
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
        },
        'lecture': {
            'syllables': ['lec', 'ture'],
            'phonemes': [['l', 'ˈɛ', 'k'], ['tʃ', 'ər']]
        },
        'female': {
            'syllables': ['fe', 'male'],
            'phonemes': [['f', 'ˈi'], ['m', 'ˌeɪ', 'l']]
        },
        'congratulations': {
            'syllables': ['con', 'gra', 'tu', 'la', 'tions'],
            'phonemes': [['k', 'ə', 'n'], ['g', 'r', 'ˌæ'], ['tʃ', 'u'], ['l', 'ˈeɪ'], ['ʃ', 'ə', 'n', 'z']]
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
    
    # 辅音组合和发音规则 - 这些组合通常保持在一起
    consonant_blends = {
        'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'sc', 'sk', 'sl', 
        'sm', 'sn', 'sp', 'st', 'sw', 'tr', 'tw', 'scr', 'spr', 'str', 'thr', 'shr', 'spl', 
        'squ', 'ch', 'sh', 'th', 'ph', 'wh', 'gn', 'kn', 'wr', 'qu', 'ck', 'dg', 'gh', 'ng', 'tch'
    }
    
    # 不可分割的发音单元 - 这些单元在音节拆分时应该保持完整
    indivisible_units = {
        'tion', 'sion', 'ture', 'cious', 'tious', 'gious', 'cial', 'tial', 'ple', 'ble', 
        'dle', 'gle', 'kle', 'fle', 'zle', 'le', 'ce', 'ge', 'se', 'ze', 'que', 'gue'
    }
    
    # 常见前缀 - 按长度排序以便优先匹配更长的前缀
    common_prefixes = [
        # 三字母及以上前缀
        'anti', 'auto', 'circum', 'contra', 'counter', 'dis', 'down', 'extra', 'hyper', 'inter', 
        'intra', 'micro', 'mid', 'mis', 'multi', 'non', 'over', 'post', 'pre', 'pro', 
        'pseudo', 'retro', 'semi', 'sub', 'super', 'supra', 'tele', 'trans', 'tri', 'ultra', 'un', 'under',
        # 双字母前缀
        'co', 'de', 'di', 'em', 'en', 'ex', 'im', 'in', 'ir', 'ob', 'of', 'on', 'op', 're',
        # 单字母前缀
        'a', 'e', 'i', 'o', 'u'
    ]
    
    # 常见后缀 - 按长度排序以便优先匹配更长的后缀
    common_suffixes = [
        # 四字母及以上后缀
        'ation', 'ition', 'cious', 'tious', 'sious', 'aceous', 'alous', 'ulous', 'ative', 'itive',
        'fully', 'ously', 'lessly', 'iness', 'ement', 'wards', 'ature', 'itive', 'tude', 'logy', 
        'graphy', 'tomy', 'metry', 'scopy', 'tion', 'sion', 'ment', 'ness', 'hood', 'ship', 'ible', 'able',
        # 三字母后缀
        'ant', 'ent', 'ary', 'ery', 'ory', 'ism', 'ist', 'ity', 'ize', 'ise', 'ify', 'ate', 'ive', 'ure', 'ous', 'ious',
        # 双字母后缀
        'al', 'ic', 'ly', 'er', 'or', 'ee', 'en', 'ed', 'ty', 'fy', 'ry', 'cy', 'is',
        # 单字母后缀
        'y', 's'
    ]
    
    # 检查单词是否以这些结尾，并且这些结尾应作为单独音节
    terminal_syllables = {
        'tion': True, 'sion': True, 'cian': True, 'le': True, 'ture': True,
        'sure': True, 'ble': True, 'dle': True, 'gle': True, 'kle': True,
        'fle': True, 'zle': True, 'cle': True, 'ple': True
    }
    
    # 1. 首先提取元音位置
    vowels = 'aeiouy'
    vowel_positions = [i for i, char in enumerate(word) if char.lower() in vowels]
    
    # 如果没有元音，则整个单词作为一个音节
    if not vowel_positions:
        return [phonemes], [word]
    
    # 2. 检查单词是否以特定后缀结尾，这些后缀应作为单独音节
    terminal_syllable = None
    terminal_start = len(word)
    
    for suffix in sorted(terminal_syllables.keys(), key=len, reverse=True):
        if word.endswith(suffix) and len(suffix) < len(word):
            terminal_syllable = suffix
            terminal_start = len(word) - len(suffix)
            break
    
    # 3. 识别前缀
    prefix = None
    prefix_end = 0
    
    for p in common_prefixes:
        if word.startswith(p) and len(p) < len(word) // 2 + 1:
            # 确保前缀不会覆盖整个单词
            if terminal_syllable and len(word) - len(p) <= len(terminal_syllable):
                continue
                
            prefix = p
            prefix_end = len(p)
            break
    
    # 4. 识别中间部分的发音单元和辅音组合
    middle = word[prefix_end:terminal_start]
    
    # 如果中间部分较短，将其作为一个整体
    if len(middle) <= 3:
        syllables = []
        if prefix:
            syllables.append(prefix)
        if middle:
            syllables.append(middle)
        if terminal_syllable:
            syllables.append(terminal_syllable)
        
        # 如果只有一个音节，则整个单词作为音节
        if not syllables:
            syllables = [word]
    else:
        # 5. 为中间部分拆分音节
        middle_syllables = []
        
        # 找出中间部分的元音位置
        middle_vowel_positions = [i - prefix_end for i in vowel_positions if prefix_end <= i < terminal_start]
        
        if len(middle_vowel_positions) <= 1:
            # 如果中间部分只有0-1个元音，将其作为一个整体
            middle_syllables = [middle]
        else:
            prev_end = 0
            for i in range(len(middle_vowel_positions) - 1):
                curr_pos = middle_vowel_positions[i]
                next_pos = middle_vowel_positions[i + 1]
                
                # 找到当前元音之后的辅音开始位置
                cons_start = curr_pos
                while cons_start + 1 < len(middle) and cons_start + 1 < next_pos and middle[cons_start + 1] in vowels:
                    cons_start += 1
                
                # 当前元音到下一个元音之间的辅音数量
                cons_count = next_pos - cons_start - 1
                boundary = 0
                
                # 根据辅音数量和特殊规则确定音节边界
                if cons_count == 0:
                    # 如果没有辅音，在元音之间分界
                    boundary = curr_pos + 1
                elif cons_count == 1:
                    # 单辅音规则：一般在辅音后分界 (V-CV)
                    # 例外：特殊的辅音-元音组合可能保持在一起
                    next_syllable = middle[cons_start+1:next_pos+1]
                    if len(next_syllable) > 1 and any(next_syllable.startswith(unit) for unit in indivisible_units):
                        # 如果下一个音节以不可分割单元开始，保留辅音在当前音节
                        boundary = cons_start + 1
                    else:
                        # 否则，标准V-CV规则
                        boundary = cons_start + 2
                else:
                    # 多辅音规则
                    # 1. 检查是否有辅音组合需要保持在一起
                    consonant_sequence = middle[cons_start+1:next_pos]
                    
                    # 检查是否有不可分割的发音单元
                    indivisible_match = False
                    for unit in sorted(indivisible_units, key=len, reverse=True):
                        if unit in consonant_sequence:
                            unit_start = consonant_sequence.find(unit)
                            # 如果不可分割单元在序列开始，保持在下一个音节
                            if unit_start == 0:
                                boundary = cons_start + 1
                            # 如果在序列结束，保持在下一个音节
                            elif unit_start + len(unit) == len(consonant_sequence):
                                boundary = cons_start + 1 + unit_start
                            # 如果在中间，在单元前分界
                            else:
                                boundary = cons_start + 1 + unit_start
                            indivisible_match = True
                            break
                    
                    # 如果没有找到不可分割单元，检查辅音组合
                    if not indivisible_match:
                        blend_match = False
                        for blend in sorted(consonant_blends, key=len, reverse=True):
                            if blend in consonant_sequence:
                                blend_start = consonant_sequence.find(blend)
                                # 如果辅音组合在序列开始，保持在下一个音节
                                if blend_start == 0:
                                    boundary = cons_start + 1
                                # 如果在序列结束，保持在下一个音节
                                elif blend_start + len(blend) == len(consonant_sequence):
                                    boundary = cons_start + 1 + blend_start
                                # 如果在中间，在组合前分界
                                else:
                                    boundary = cons_start + 1 + blend_start
                                blend_match = True
                                break
                        
                        # 如果没有找到辅音组合，在辅音中间分界
                        if not blend_match:
                            boundary = cons_start + 1 + cons_count // 2
                
                if boundary > 0 and boundary < len(middle):
                    middle_syllables.append(middle[prev_end:boundary])
                    prev_end = boundary
            
            # 添加最后一个音节
            if prev_end < len(middle):
                middle_syllables.append(middle[prev_end:])
        
        # 6. 组合前缀、中间音节和终止音节
        syllables = []
        if prefix:
            syllables.append(prefix)
        syllables.extend(middle_syllables)
        if terminal_syllable:
            syllables.append(terminal_syllable)
    
    # 7. 修正音节边界，确保每个音节至少有一个元音
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
    
    # 8. 连续优化：合并过短的音节
    i = 0
    while i < len(final_syllables) - 1:
        curr_syl = final_syllables[i]
        next_syl = final_syllables[i+1]
        
        # 如果当前音节过短（1-2个字符）并且不是元音开头的音节
        if len(curr_syl) <= 2 and not any(curr_syl.startswith(v) for v in vowels):
            # 检查是否应该与下一个音节合并
            merged = False
            
            # 特殊规则：如果是以元音开头的前缀（如a-, e-, i-），保持独立
            if curr_syl in ['a', 'e', 'i', 'o', 'u'] and i == 0:
                i += 1
                continue
                
            # 如果当前和下一个音节组合起来形成一个有意义的单元，则合并
            combined = curr_syl + next_syl
            if any(combined.startswith(p) for p in common_prefixes) or any(combined.endswith(s) for s in common_suffixes):
                final_syllables[i] = combined
                final_syllables.pop(i+1)
                merged = True
            
            # 如果没有基于语义合并，检查长度和位置
            if not merged and len(curr_syl) == 1 and curr_syl not in vowels:
                # 单个辅音字母通常合并到下一个音节
                final_syllables[i+1] = curr_syl + next_syl
                final_syllables.pop(i)
                continue  # 不增加i，因为我们删除了当前元素
        
        i += 1
    
    # 9. 确认最终的音节集不会遗漏或重复字母
    joined = ''.join(final_syllables)
    if joined != word:
        # 如果有问题，回退到基于元音的简单拆分
        final_syllables = []
        syllable_start = 0
        
        # 根据元音位置确定音节
        for i in range(len(vowel_positions)):
            vowel_pos = vowel_positions[i]
            
            # 第一个元音的情况
            if i == 0:
                # 如果元音不是第一个字符，包括前面的辅音
                if vowel_pos > 0:
                    final_syllables.append(word[:vowel_pos+1])
                    syllable_start = vowel_pos + 1
                else:
                    # 如果元音是第一个字符，只添加元音
                    final_syllables.append(word[0])
                    syllable_start = 1
            else:
                # 确定音节边界
                prev_vowel = vowel_positions[i-1]
                
                # 计算两个元音之间的辅音数量
                cons_count = vowel_pos - prev_vowel - 1
                
                if cons_count == 0:
                    # 两个连续元音，在第一个元音后分界
                    if syllable_start <= prev_vowel:
                        final_syllables.append(word[syllable_start:prev_vowel+1])
                        syllable_start = prev_vowel + 1
                elif cons_count == 1:
                    # 一个辅音，通常在元音后分界 (V-CV)
                    if syllable_start <= prev_vowel + 1:
                        final_syllables.append(word[syllable_start:prev_vowel+2])
                        syllable_start = prev_vowel + 2
                else:
                    # 多个辅音，根据位置确定边界
                    boundary = prev_vowel + 1 + cons_count // 2
                    if syllable_start <= boundary:
                        final_syllables.append(word[syllable_start:boundary])
                        syllable_start = boundary
        
        # 添加最后一部分
        if syllable_start < len(word):
            final_syllables.append(word[syllable_start:])
    
    # 10. 为音节分配音标
    syllable_phonemes = []
    
    if len(final_syllables) == 1:
        # 单音节单词，所有音标分配给整个单词
        syllable_phonemes = [phonemes]
    else:
        # 尝试基于音素边界分配音标
        # 简单方案：根据音节长度比例分配音标
        total_length = sum(len(s) for s in final_syllables)
        
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
    
    logger.info(f"音节拆分（人类习惯版）: {word} -> {final_syllables}")
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