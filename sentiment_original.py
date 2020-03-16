from typing import TextIO, List, Union, Dict, Tuple

# PART I: File I/O, strings, lists

def is_word(token: str) -> bool:
    '''Return True IFF token is an alphabetic word optionally containing
    forward slashes or dashes.
    
    >>> is_word('Amazing')
    True
    >>> is_word('writer/director')
    True
    >>> is_word('true-to-life')
    True
    >>> is_word("'re")
    False
    >>> is_word("1960s")
    False
    '''
    is_valid = True
    for char in token:
        if not char.isalpha() and char != '/' and char != '-':
            is_valid = False
    return is_valid


def get_word_list(statement: str) -> List[str]:
    '''Return a list of words contained in statement, converted to lowercase. 
    Use is_word to determine whether each token in statement is a word.
    
    >>> get_word_list('A terrible , 1970s mess of true-crime nonsense from writer/director Shyamalan .')
    ['a', 'terrible', 'mess', 'of', 'true-crime', 'nonsense', 'from', 'writer/director', 'shyamalan']
    '''
    
    split_lst = statement.lower().strip().split(' ')
    return_lst = split_lst.copy()
    
    for item in split_lst:
        if not is_word(item):
            return_lst.remove(item)
    return return_lst
            
    

def judge(score: float) -> str:
    '''Return 'negative' if score is 1.5 or less.
    Return 'positive' if score is 2.5 or more.
    Return 'neutral' otherwise.
    >>> judge(1.3)
    'negative'
    >>> judge(1.8)
    'neutral'
    >>> judge(3.4)
    'positive'
    '''
    
    if score <= 1.5:
        return 'negative'
    if score >= 2.5:
        return 'positive'
    return 'neutral'



def word_kss_scan(word: str, file: TextIO) -> Union[None, float]:
    '''Given file composed of rated movie reviews, return the average score
    of all occurrences of word in file. If word does not occur in file, return None.
    [examples not required]
    '''
    
    score = 0.0
    count = 0
    line = file.readline()
    check_lst = get_word_list(line)
   
    while line != '':
        for char in check_lst:
            if word == char:
                count += 1
                score += float(line[0])
        line = file.readline()
        check_lst = get_word_list(line)
    
    if count != 0:
        return score/count
    return None
    
    



# PART II: Dictionaries 

def extract_kss(file: TextIO) -> Dict[str, List[int]]:
    '''Given file composed of rated movie reviews, return a dictionary
    containing all words in file as keys. For each key, store a list
    containing the total sum of review scores and the number of times
    the key has occurred as a value, e.g., { 'a' : [12, 4] }
    [examples not required]
    
    '''
    return_dict = {}
    line = file.readline()
    key_lst = get_word_list(line)
    
    while line != '':
        for item in key_lst:
            if item not in return_dict:
                count = 1
                score = int(line[0])
                return_dict[item] = [score]
                return_dict[item].append(count)
            else:
                return_dict[item][0] += int(line[0])
                return_dict[item][1] += 1
            
        line = file.readline()
        key_lst = get_word_list(line)
        
    return return_dict
    
    
    


def word_kss(word: str, kss: Dict[str, List[int]]) -> Union[float, None]:
    '''Return the Known Sentiment Score of word if it appears in kss. 
    If word does not appear in kss, return None.
    [examples not required]
    '''    
    word = word.lower()
    if word in kss:
        return  kss[word][0] / kss[word][1]
    return None
             
             
def statement_pss(statement: str, kss: Dict[str, List[int]]) -> Union[float, None]:
    '''Return the Predicted Sentiment Score of statement based on
    word Known Sentiment Scores from kss.
    Return None if statement contains no words from kss.
    '''
    
    word_lst = get_word_list(statement)
    avg_score = 0.0
    num = 0
    for item in word_lst:
        if item in kss:
            avg_score += word_kss(item, kss)
            num += 1
    
    if num != 0:
        return avg_score / num
    return None
    
    




# PART III: Word Frequencies

def score(item: Tuple[str, List[int]]) -> float:
    '''Given item as a (key, value) tuple, return the
    ratio of the first and second integer in value
    '''
    
    return item[1][0] / item[1][1]
    


def most_extreme_words(count: int, min_occ: int, 
                       kss: Dict[str, List[int]], 
                       pos: bool):
    '''Return a list of lists containing the count most extreme words
    that occur at least min_occ times in kss.
    Each item in the list is formatted as follows:
    [word, average score, number of occurrences]
    If pos is True, return the most positive words.
    If pos is False, return the most negative words.
    [examples not required]
    '''
    kss_tuples = kss.items()
    if pos:
        sorted_kss = sorted(kss_tuples, key = score, reverse = True)
    else:
        sorted_kss = sorted(kss_tuples, key = score, reverse = True)
        
    mew_lst = []
    num = 0
    
    while sorted_kss != [] and num < count:
        tup = sorted_kss.pop(0)
        if tup[1][1] >= min_occ:
            mew_lst.append([tup[0], score(tup), tup[1][1]])
            num += 1
    return mew_lst
        
    
    
    
def most_negative_words(count: int, min_occ: int, 
                        kss: Dict[str, List[int]]):
    '''Return a list of the count most negative words that 
    occur at least min_occ times in kss.
    '''
    
    return most_extreme_words(count, min_occ, kss, False) 
    
def most_positive_words(count: int, min_occ: int,
                        kss: Dict[str, List[int]]):
    '''Return a list of the count most positive words that
    occur at least min_occ times in kss.
    '''
    return most_extreme_words(count, min_occ, kss, True)
    
    

        
    
if __name__ == "__main__":

# Pick a dataset    
    #dataset = 'tiny.txt'
    #dataset = 'small.txt'
    #dataset = 'medium.txt'
    #dataset = 'full.txt'
    
    # Your test code here
    pass