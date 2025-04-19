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
        # 获取原始CMU音标
        phonemes = d[word][0]
        # 将CMU音标转换为IPA音标
        cmu_to_ipa = {
            'HH': 'h',    # h
            'AH': 'ə',    # ə
            'N': 'n',     # n
            'T': 't',     # t
            'ER': 'ɝ',    # ɝ
            'AE': 'æ',    # æ
            'K': 'k',     # k
            'CH': 'tʃ',   # tʃ
            'W': 'w',     # w
            'L': 'l',     # l
            'IY': 'i',    # i
            'EH': 'ɛ',    # ɛ
            'R': 'r',     # r
            'IH': 'ɪ',    # ɪ
            'OW': 'o',    # o
            'AY': 'aɪ',   # aɪ
            'EY': 'eɪ',   # eɪ
            'OY': 'ɔɪ',   # ɔɪ
            'AW': 'aʊ',   # aʊ
            'UW': 'u',    # u
            'NG': 'ŋ',    # ŋ
            'SH': 'ʃ',    # ʃ
            'TH': 'θ',    # θ
            'DH': 'ð',    # ð
            'ZH': 'ʒ',    # ʒ
            'JH': 'dʒ',   # dʒ
            'Y': 'j',     # j
        }
        
        # 转换音标并移除重音标记
        mapped_phonemes = []
        for phoneme in phonemes:
            # 移除重音标记
            base_phoneme = re.sub(r'\d+', '', phoneme)
            # 转换音标
            ipa = cmu_to_ipa.get(base_phoneme, base_phoneme)
            mapped_phonemes.append(ipa)
        
        return mapped_phonemes
    return None

def split_syllables(word):
    """基于音标和常见英语单词结构拆分音节"""
    word = word.lower()
    # 获取音标
    phonemes = get_phonetic(word)
    if not phonemes:
        return None, None
    
    # 初始化常见的英语单词结构（前缀、后缀）
    common_suffixes = {
        'ture': ['tʃ', 'ɝ'],      # lecture, nature
        'tion': ['ʃ', 'ə', 'n'],   # nation, station
        'sion': ['ʒ', 'ə', 'n'],   # vision, fusion
        'cious': ['ʃ', 'ə', 's'],  # delicious
        'tious': ['ʃ', 'ə', 's'],  # cautious
        'cial': ['ʃ', 'ə', 'l'],   # social
        'tial': ['ʃ', 'ə', 'l'],   # partial
        'er': ['ɝ'],               # hunter, paper
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
        'ex': ['ɪ', 'k', 's'],     # exchange, example
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
        'over': ['o', 'v', 'ɝ'],   # overcome, overlook
        'under': ['ʌ', 'n', 'd', 'ɝ'], # understand, underestimate
        'sub': ['s', 'ʌ', 'b'],    # submarine, submerge
        'super': ['s', 'u', 'p', 'ɝ'], # supermarket, superpower
        'inter': ['ɪ', 'n', 't', 'ɝ'], # international, interrupt
        'anti': ['æ', 'n', 't', 'i'], # antibiotic, antifreeze
    }
    
    # 这些是单音节元音前缀
    vowel_prefixes = ['a', 'e', 'i', 'o', 'u']
    
    # 强制处理特定单词（作为回退方案，确保最常见的问题单词正确处理）
    forced_patterns = {
        'hunter': ['hun', 'ter'],
        'paper': ['pa', 'per'],
        'water': ['wa', 'ter'],
        'letter': ['let', 'ter'],
        'exchange': ['ex', 'change'],
        'extend': ['ex', 'tend'],
        'explain': ['ex', 'plain'],
        'export': ['ex', 'port'],
        'exclude': ['ex', 'clude'],
        'lecture': ['lec', 'ture'],
        'creature': ['crea', 'ture'],
        'furniture': ['fur', 'ni', 'ture'],
        'actually': ['ac', 'tu', 'al', 'ly'],
        'nationality': ['na', 'tion', 'a', 'li', 'ty'],
        'agree': ['a', 'gree']
    }
    
    # 先检查是否是强制处理的单词
    if word in forced_patterns:
        logger.info(f"特殊处理单词: {word} -> {forced_patterns[word]}")
        
        # 为每个音节分配音标
        syllable_phonemes = []
        remaining_phonemes = phonemes.copy()
        
        for syllable in forced_patterns[word]:
            # 根据音节长度分配音标
            phoneme_count = min(len(syllable), len(remaining_phonemes))
            syllable_phonemes.append(remaining_phonemes[:phoneme_count])
            remaining_phonemes = remaining_phonemes[phoneme_count:]
        
        # 如果还有剩余音标，添加到最后一个音节
        if remaining_phonemes:
            syllable_phonemes[-1].extend(remaining_phonemes)
            
        return syllable_phonemes, forced_patterns[word]
    
    # 使用CMU音标数据进行分析
    if word not in d:
        return None, None
        
    # 获取原始CMU音标
    cmu_phonemes = d[word][0]
    logger.info(f"处理单词: {word}, CMU音标: {cmu_phonemes}")
    
    # 找出所有元音音标的位置
    vowel_indices = []
    for i, phoneme in enumerate(cmu_phonemes):
        # 在CMU音标中，元音音标通常包含数字（表示重音）
        if any(phoneme.startswith(v) for v in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']):
            vowel_indices.append(i)
    
    logger.info(f"元音索引: {vowel_indices}")
    
    # 检查特定单词模式
    # 检查以'er'结尾的单词，如 hunter
    if word.endswith('er') and len(word) > 2 and len(vowel_indices) >= 2:
        # 确保最后的'er'是一个单独的音节
        # 通常，CMU音标中的'ER'会是最后一个元音
        if 'ER' in cmu_phonemes[-2:]:
            # 找到'er'的起始位置
            er_pos = word.rfind('er')
            if er_pos > 0:
                prefix = word[:er_pos]
                
                # 使用pyphen拆分前缀
                prefix_hyphenated = dic.inserted(prefix)
                prefix_syllables = prefix_hyphenated.split('-')
                
                # 组合前缀音节和'er'
                result_syllables = prefix_syllables + ['er']
                
                # 分配音标
                syllable_phonemes = []
                phonemes_per_syllable = len(phonemes) // len(result_syllables)
                
                for i in range(len(result_syllables) - 1):
                    start = i * phonemes_per_syllable
                    end = (i + 1) * phonemes_per_syllable
                    syllable_phonemes.append(phonemes[start:end])
                
                # 最后一个音节（'er'）包含剩余音标
                syllable_phonemes.append(phonemes[(len(result_syllables) - 1) * phonemes_per_syllable:])
                
                logger.info(f"特殊处理'er'后缀: {result_syllables}, 音标: {syllable_phonemes}")
                return syllable_phonemes, result_syllables
    
    # 检查以'ex'开头的单词，如 exchange
    if word.startswith('ex') and len(word) > 2:
        remainder = word[2:]
        
        # 使用pyphen拆分剩余部分
        remainder_hyphenated = dic.inserted(remainder)
        remainder_syllables = remainder_hyphenated.split('-')
        
        # 组合'ex'和剩余音节
        result_syllables = ['ex'] + remainder_syllables
        
        # 分配音标
        syllable_phonemes = []
        # 通常'ex'对应前两个音标
        ex_phonemes = phonemes[:min(3, len(phonemes))]
        remainder_phonemes = phonemes[min(3, len(phonemes)):]
        
        syllable_phonemes.append(ex_phonemes)
        
        # 为剩余部分均匀分配音标
        if remainder_syllables:
            phonemes_per_syllable = len(remainder_phonemes) // len(remainder_syllables)
            
            for i in range(len(remainder_syllables) - 1):
                start = i * phonemes_per_syllable
                end = (i + 1) * phonemes_per_syllable
                syllable_phonemes.append(remainder_phonemes[start:end])
            
            # 最后一个音节包含剩余音标
            syllable_phonemes.append(remainder_phonemes[(len(remainder_syllables) - 1) * phonemes_per_syllable:])
        
        logger.info(f"特殊处理'ex'前缀: {result_syllables}, 音标: {syllable_phonemes}")
        return syllable_phonemes, result_syllables
    
    # 检查以'ture'结尾的单词，如 lecture
    if word.endswith('ture') and len(word) > 4:
        # 找到'ture'的起始位置
        ture_pos = word.rfind('ture')
        if ture_pos > 0:
            prefix = word[:ture_pos]
            
            # 使用pyphen拆分前缀
            prefix_hyphenated = dic.inserted(prefix)
            prefix_syllables = prefix_hyphenated.split('-')
            
            # 组合前缀音节和'ture'
            result_syllables = prefix_syllables + ['ture']
            
            # 分配音标
            syllable_phonemes = []
            # 尝试确定'ture'的音标位置
            ture_phoneme_index = len(phonemes) - 2  # 通常'ture'对应最后2-3个音标
            
            # 为前缀部分分配音标
            phonemes_per_prefix_syllable = ture_phoneme_index // len(prefix_syllables)
            
            for i in range(len(prefix_syllables)):
                start = i * phonemes_per_prefix_syllable
                end = (i + 1) * phonemes_per_prefix_syllable if i < len(prefix_syllables) - 1 else ture_phoneme_index
                syllable_phonemes.append(phonemes[start:end])
            
            # 为'ture'分配音标
            syllable_phonemes.append(phonemes[ture_phoneme_index:])
            
            logger.info(f"特殊处理'ture'后缀: {result_syllables}, 音标: {syllable_phonemes}")
            return syllable_phonemes, result_syllables
    
    # 分析单词是否包含常见后缀
    suffix_result = None
    remaining_word = word
    suffix_phonemes = []
    
    # 从最长的后缀开始检查
    for suffix, suffix_ipa in sorted(common_suffixes.items(), key=lambda x: len(x[0]), reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix):
            # 确保suffix不是整个单词
            remaining_word = word[:-len(suffix)]
            suffix_phonemes = suffix_ipa
            suffix_result = suffix
            logger.info(f"找到后缀: {suffix} in {word}")
            break
    
    # 分析单词是否以元音字母开头后跟辅音
    prefix_result = None
    prefix_phonemes = []
    
    # 检查常见前缀
    for prefix, prefix_ipa in sorted(common_prefixes.items(), key=lambda x: len(x[0]), reverse=True):
        if remaining_word.startswith(prefix) and len(remaining_word) > len(prefix):
            prefix_result = prefix
            remaining_word = remaining_word[len(prefix):]
            prefix_phonemes = prefix_ipa
            logger.info(f"找到前缀: {prefix} in {word}")
            break
    
    # 如果没有找到常见前缀，检查单个元音前缀
    if not prefix_result:
        if any(remaining_word.startswith(prefix) for prefix in vowel_prefixes) and len(remaining_word) > 1 and remaining_word[1] not in 'aeiou':
            # 单个元音字母开头后跟辅音，如"a-gree"
            prefix_result = remaining_word[0]
            remaining_word = remaining_word[1:]
            # 找到对应的音标（通常是第一个元音音标）
            if vowel_indices and vowel_indices[0] < len(phonemes):
                prefix_phonemes = [phonemes[vowel_indices[0]]]
            logger.info(f"找到单音节前缀: {prefix_result} in {word}")
    
    # 如果找到前缀和后缀，并且剩余部分是一个合理的单词片段
    if (suffix_result or prefix_result) and len(remaining_word) > 0:
        parts = []
        phoneme_parts = []
        
        # 添加前缀（如果有）
        if prefix_result:
            parts.append(prefix_result)
            phoneme_parts.append(prefix_phonemes)
        
        # 使用pyphen拆分剩余部分
        hyphenated = dic.inserted(remaining_word)
        remaining_parts = hyphenated.split('-')
        
        # 估算剩余部分的音标
        remaining_phonemes = [] 
        if prefix_result and suffix_result:
            # 如果有前缀和后缀，尝试从中间取音标
            start_idx = len(prefix_phonemes) if prefix_phonemes else 0
            end_idx = len(phonemes) - len(suffix_phonemes) if suffix_phonemes else len(phonemes)
            remaining_phonemes = phonemes[start_idx:end_idx]
        elif prefix_result:
            # 如果只有前缀，从前缀后面取音标
            start_idx = len(prefix_phonemes) if prefix_phonemes else 0
            remaining_phonemes = phonemes[start_idx:]
        elif suffix_result:
            # 如果只有后缀，从开头取到后缀前
            end_idx = len(phonemes) - len(suffix_phonemes) if suffix_phonemes else len(phonemes)
            remaining_phonemes = phonemes[:end_idx]
        
        # 根据音节数量均匀分配音标
        if remaining_parts and remaining_phonemes:
            # 根据每个音节的长度比例分配音标
            total_length = sum(len(part) for part in remaining_parts)
            phoneme_distribution = []
            
            current_idx = 0
            for part in remaining_parts:
                # 按比例分配音标
                part_ratio = len(part) / total_length
                phoneme_count = max(1, round(part_ratio * len(remaining_phonemes)))
                # 确保不超出范围
                phoneme_count = min(phoneme_count, len(remaining_phonemes) - current_idx)
                
                part_phonemes = remaining_phonemes[current_idx:current_idx + phoneme_count]
                phoneme_distribution.append(part_phonemes)
                current_idx += phoneme_count
            
            # 如果还有剩余音标，添加到最后一个部分
            if current_idx < len(remaining_phonemes):
                phoneme_distribution[-1].extend(remaining_phonemes[current_idx:])
            
            # 添加剩余部分及其音标
            parts.extend(remaining_parts)
            phoneme_parts.extend(phoneme_distribution)
        else:
            # 如果无法分配，直接添加剩余部分
            parts.append(remaining_word)
            phoneme_parts.append(remaining_phonemes)
        
        # 添加后缀（如果有）
        if suffix_result:
            parts.append(suffix_result)
            phoneme_parts.append(suffix_phonemes)
        
        # 返回结果
        logger.info(f"基于结构拆分: {parts}, 音标: {phoneme_parts}")
        return phoneme_parts, parts
    
    # 如果没有识别出特殊结构，使用音标位置确定音节边界
    syllable_boundaries = []
    
    # 根据元音位置和辅音规则确定音节边界
    for i in range(len(vowel_indices) - 1):
        current_vowel_pos = vowel_indices[i]
        next_vowel_pos = vowel_indices[i + 1]
        consonant_count = next_vowel_pos - current_vowel_pos - 1
        
        if consonant_count == 0:
            # 如果没有辅音，边界在两个元音之间
            syllable_boundaries.append(current_vowel_pos + 1)
        elif consonant_count == 1:
            # 如果有一个辅音，边界在元音和辅音之间
            syllable_boundaries.append(current_vowel_pos + 1)
        elif consonant_count >= 2:
            # 如果有多个辅音，通常在辅音中间分界
            syllable_boundaries.append(current_vowel_pos + 1 + consonant_count // 2)
    
    # 确定CMU音标的音节
    cmu_syllables = []
    start_idx = 0
    for boundary in syllable_boundaries:
        cmu_syllables.append(cmu_phonemes[start_idx:boundary])
        start_idx = boundary
    cmu_syllables.append(cmu_phonemes[start_idx:])
    
    # 将CMU音标音节转换为IPA音标音节
    syllable_phonemes = []
    for cmu_syllable in cmu_syllables:
        ipa_syllable = []
        for phoneme in cmu_syllable:
            # 移除重音标记
            base_phoneme = re.sub(r'\d+', '', phoneme)
            # 转换音标
            cmu_to_ipa = {
                'HH': 'h',    # h
                'AH': 'ə',    # ə
                'N': 'n',     # n
                'T': 't',     # t
                'ER': 'ɝ',    # ɝ
                'AE': 'æ',    # æ
                'K': 'k',     # k
                'CH': 'tʃ',   # tʃ
                'W': 'w',     # w
                'L': 'l',     # l
                'IY': 'i',    # i
                'EH': 'ɛ',    # ɛ
                'R': 'r',     # r
                'IH': 'ɪ',    # ɪ
                'OW': 'o',    # o
                'AY': 'aɪ',   # aɪ
                'EY': 'eɪ',   # eɪ
                'OY': 'ɔɪ',   # ɔɪ
                'AW': 'aʊ',   # aʊ
                'UW': 'u',    # u
                'NG': 'ŋ',    # ŋ
                'SH': 'ʃ',    # ʃ
                'TH': 'θ',    # θ
                'DH': 'ð',    # ð
                'ZH': 'ʒ',    # ʒ
                'JH': 'dʒ',   # dʒ
                'Y': 'j',     # j
            }
            ipa = cmu_to_ipa.get(base_phoneme, base_phoneme)
            ipa_syllable.append(ipa)
        syllable_phonemes.append(ipa_syllable)
    
    # 使用pyphen进行初步音节拆分，作为参考
    hyphenated = dic.inserted(word)
    initial_syllables = hyphenated.split('-')
    
    # 特殊情况：如果音标音节数量和pyphen拆分音节数量相同，使用pyphen结果
    if len(syllable_phonemes) == len(initial_syllables):
        logger.info(f"使用pyphen拆分: {initial_syllables}, 音标: {syllable_phonemes}")
        return syllable_phonemes, initial_syllables
    
    # 否则，使用更复杂的映射算法
    # 根据单词结构拆分
    # 这里参考辅音-元音组合的常见规则
    vowels = 'aeiouy'
    consonants = 'bcdfghjklmnpqrstvwxz'
    
    syllables = []
    i = 0
    while i < len(word):
        # 如果以常见后缀结尾，单独处理
        found_suffix = False
        for suffix in sorted(common_suffixes.keys(), key=len, reverse=True):
            if word[i:].endswith(suffix) and i + len(suffix) < len(word):
                # 找到后缀
                if i > 0:
                    # 添加前面的部分
                    syllables.append(word[:i])
                # 添加后缀
                syllables.append(word[i:])
                found_suffix = True
                i = len(word)  # 结束循环
                break
        
        if found_suffix:
            break
            
        # 如果当前位置是元音
        if i < len(word) and word[i] in vowels:
            # 找到下一个元音的位置
            next_vowel_pos = i + 1
            while next_vowel_pos < len(word) and word[next_vowel_pos] not in vowels:
                next_vowel_pos += 1
            
            # 如果没有找到下一个元音，添加剩余部分
            if next_vowel_pos >= len(word):
                syllables.append(word[i:])
                break
                
            # 如果元音之间有超过一个辅音，应该在辅音中间分界
            consonant_count = next_vowel_pos - i - 1
            if consonant_count > 1:
                # 在辅音中间分界
                split_pos = i + 1 + consonant_count // 2
                syllables.append(word[i:split_pos])
                i = split_pos
            else:
                # 只有一个辅音，加到元音后面
                syllables.append(word[i:i+2])
                i += 2
        else:
            # 如果是辅音开头，找到第一个元音
            vowel_pos = i
            while vowel_pos < len(word) and word[vowel_pos] not in vowels:
                vowel_pos += 1
                
            if vowel_pos >= len(word):
                # 如果没有找到元音，添加剩余部分
                syllables.append(word[i:])
                break
                
            # 找到元音后的下一个辅音
            next_consonant_pos = vowel_pos + 1
            while next_consonant_pos < len(word) and word[next_consonant_pos] in vowels:
                next_consonant_pos += 1
                
            if next_consonant_pos >= len(word):
                # 如果没有找到辅音，添加剩余部分
                syllables.append(word[i:])
                break
                
            # 添加到元音后的第一个辅音
            syllables.append(word[i:next_consonant_pos])
            i = next_consonant_pos
    
    # 将生成的音节与音标匹配
    if len(syllables) == len(syllable_phonemes):
        logger.info(f"使用结构拆分: {syllables}, 音标: {syllable_phonemes}")
        return syllable_phonemes, syllables
    
    # 如果音节数量不匹配，按照元音位置重新拆分
    # 这种情况下优先使用pyphen结果
    if len(initial_syllables) == len(syllable_phonemes):
        logger.info(f"回退到pyphen拆分: {initial_syllables}, 音标: {syllable_phonemes}")
        return syllable_phonemes, initial_syllables
        
    # 最后的回退方案：将单词均匀拆分为指定数量的音节
    # 根据元音位置拆分
    even_syllables = []
    vowel_positions = [i for i, c in enumerate(word) if c in vowels]
    
    if not vowel_positions:
        # 如果没有元音，返回整个单词
        logger.info(f"没有找到元音，返回整个单词: {word}")
        return [phonemes], [word]
        
    # 将元音均匀分组
    vowels_per_syllable = len(vowel_positions) // len(syllable_phonemes)
    if vowels_per_syllable < 1:
        vowels_per_syllable = 1
        
    # 根据元音位置拆分
    for i in range(0, len(vowel_positions), vowels_per_syllable):
        if i + vowels_per_syllable >= len(vowel_positions) or len(even_syllables) == len(syllable_phonemes) - 1:
            # 最后一个音节包含所有剩余元音
            end_pos = len(word)
        else:
            # 在两个元音组之间的中点拆分
            next_group_start = vowel_positions[i + vowels_per_syllable]
            end_vowel = vowel_positions[i + vowels_per_syllable - 1]
            
            # 找到这个元音后的第一个辅音
            end_pos = end_vowel + 1
            while end_pos < len(word) and word[end_pos] in vowels:
                end_pos += 1
                
            # 如果有多个连续辅音，在中间拆分
            consonant_count = 0
            temp_pos = end_pos
            while temp_pos < len(word) and temp_pos < next_group_start and word[temp_pos] not in vowels:
                consonant_count += 1
                temp_pos += 1
                
            if consonant_count > 1:
                end_pos += consonant_count // 2
        
        # 如果不是第一个音节，从当前位置开始
        start_pos = 0 if not even_syllables else even_syllables[-1][1]
        even_syllables.append((start_pos, end_pos))
        
        # 如果已经达到所需的音节数量，结束循环
        if len(even_syllables) == len(syllable_phonemes):
            break
    
    # 将位置转换为实际音节
    result_syllables = []
    for start, end in even_syllables:
        result_syllables.append(word[start:end])
        
    # 确保所有字母都被包含
    if even_syllables and even_syllables[-1][1] < len(word):
        result_syllables[-1] += word[even_syllables[-1][1]:]
        
    # 如果生成的音节数量仍然不匹配，使用最简单的方法：均匀拆分
    if len(result_syllables) != len(syllable_phonemes):
        # 按字符数均匀拆分
        chars_per_syllable = len(word) // len(syllable_phonemes)
        result_syllables = []
        
        for i in range(len(syllable_phonemes) - 1):
            start = i * chars_per_syllable
            end = (i + 1) * chars_per_syllable
            result_syllables.append(word[start:end])
            
        # 最后一个音节包含剩余字符
        result_syllables.append(word[(len(syllable_phonemes) - 1) * chars_per_syllable:])
    
    logger.info(f"使用均匀拆分: {result_syllables}, 音标: {syllable_phonemes}")
    return syllable_phonemes, result_syllables

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
        
        result = {
            'word': word,
            'full_phonetic': phonemes,
            'syllables': [{'text': s, 'phonetic': p} for s, p in zip(syllables, syllable_phonemes)]
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