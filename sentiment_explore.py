from typing import TextIO, List, Union, Dict, Tuple
from sentiment import *

import random
import nltk
import numpy as np
import os, sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize,WordPunctTokenizer
from nltk.stem import WordNetLemmatizer

# Stoplists function
def delete_stoplist(dataset: list, stoplist: list) -> list:
    '''This function returns a list that removes the words that appear in the
    list of all stoplists from dataset list.
    >>>delete_stoplist(['It is a raining day!', 'This is a good dog.'],
                       ['a', 'is'])
    ['It raining day ! ', 'This good dog . ']
    '''
    output_list = []
    for review in dataset:
        review_list = word_tokenize(review)
        #print(review_list)
        output_str = ''
        for word in review_list:
            if word.lower() not in stoplist:
                output_str += word + ' '
        output_list.append(output_str)
    return output_list


# Stemmer function
def tackle_stemmer(dataset: list, method: str) -> list:
    '''Precondition: method are 'p', 'l', or 's'.
    'p' means porter method, 'l' means lancaster method, 
    's' means snowball method.
    This function extracts stems from all the words in the dataset list 
    according to different rules and returns a new list.
    >>>tackle_stemmer(['It is a raining day!', 'This is a good dog.'],'p')
    ['It is a rain day ! ', 'thi is a good dog . ']
    >>>tackle_stemmer(['It is a raining day!', 'This is a good dog.'],'l')
    ['it is a rain day ! ', 'thi is a good dog . ']
    >>>tackle_stemmer(['It is a raining day!', 'This is a good dog.'],'s')
    ['it is a rain day ! ', 'this is a good dog . ']
    '''
    porter =PorterStemmer()
    lancaster = LancasterStemmer()
    snowball= SnowballStemmer('english')   
    output_list = []
    
    if method == 'p':
        stemmer = porter
    elif method == 'l':
        stemmer = lancaster
    elif method == 's':
        stemmer =snowball
        
    for review in dataset:
        review_list = word_tokenize(review)
        output_str = ''
        for word in review_list:
            output_str += stemmer.stem(word) + ' '
        output_list.append(output_str)        
    return output_list


# lemmatization function
def tackle_lemma(dataset: list, method: str) -> list:
    '''Precondition: method is 'n' or 'v'.
    'n' means noun lemmatizaer, 'v' means verb lemmatizer.
    This function performs a morphological restoration of all the words in the 
    dataset list according to different rules and returns a new list.
    >>>tackle_lemma(['It is a raining day!', 'This is a good dog.'],'n')
    ['It is a raining day ! ', 'This is a good dog . ']
    >>>tackle_lemma(['It is a raining day!', 'This is a good dog.'],'v')
    ['It be a rain day ! ', 'This be a good dog . ']
    '''
    lemmatizer = WordNetLemmatizer()  
    output_list = []
        
    for review in dataset:
        review_list = word_tokenize(review)
        output_str = ''
        for word in review_list:
            output_str += lemmatizer.lemmatize(word, method) + ' '
        output_list.append(output_str)        
    return output_list


# lemmatization function with POS Tag
def tackle_lemma_2(dataset: list) -> list:
    '''This function makes a part-of-speech judgment of all the words in the 
    dataset list, and then uses the different rules to perform the
    morphological restoration according to its part of speech, and returns to 
    generate a new list.
    >>>tackle_lemma_2(['It is a raining day!', 'This is a good dog.'])
    ['It be a raining day ! ', 'This be a good dog . ']
    '''
    lemmatizer = WordNetLemmatizer()  
    output_list = []
        
    for review in dataset:
        output_str = ''       
        for word, tag in nltk.pos_tag(word_tokenize(review)):
            if tag.startswith('NN'):
                output_str += lemmatizer.lemmatize(word, pos='n') + ' '
            elif tag.startswith('VB'):
                output_str += lemmatizer.lemmatize(word, pos='v') + ' '
            elif tag.startswith('JJ'):
                output_str += lemmatizer.lemmatize(word, pos='a') + ' '
            elif tag.startswith('R'):
                output_str += lemmatizer.lemmatize(word, pos='r') + ' '
            else:
                output_str += word + ' '          
        output_list.append(output_str)        
    return output_list


def output_result(test_set: list, kss: Dict[str, List[int]]):
    '''Note: this function contains print(), which only use to print the 
    statistical result.
    This function return the confusion matrix, accurancy value and summary 
    table according to test dataset and kss we draw from the training data
    set.
    '''
    #print(true_score, pre_score)
    t_score = []
    p_score =[]       
    for item in test_set:
        t_score.append(judge(float(item[0])))
        if statement_pss(item, kss) != None: # aviod the NoneType error
            p_score.append(judge(float(statement_pss(item, kss))))
        else:
            p_score.append('fail')
        
    # output the result
    true_score = []
    pre_score = []
    for index in range(len(p_score)):
        if p_score[index] != 'fail':
            true_score.append(t_score[index]) 
            pre_score.append(p_score[index])

    cm = confusion_matrix(true_score, pre_score, labels = 
                          ['negative', 'neutral', 'positive'])
    ac = accuracy_score(true_score, pre_score, normalize=True, 
                        sample_weight=None)    
    target_names = ['negative', 'neutral', 'positive']
    print(cm)
    print(classification_report(true_score, pre_score, 
                                target_names = target_names)) 


if __name__ == "__main__":

    # Pick a dataset    
    dataset_ti = 'tiny.txt'
    dataset_sm = 'small.txt'
    dataset_me = 'medium.txt'
    dataset_fu = 'full.txt'
    
    # prepare work
    with open(dataset_fu, 'r') as file:
        fu_set = file.readlines()
    with open(dataset_fu, 'r') as file:
        fu_kss = extract_kss(file)    
    with open(dataset_me, 'r') as file:
        me_set = file.readlines()    
    with open(dataset_me, 'r') as file:
        me_kss = extract_kss(file)     
    with open(dataset_sm, 'r') as file: # use it as test dataset
        sm_set = file.readlines()  
    with open('.\most_common_english_words.txt', 'r') as stoplist_file:
        stoplist = []
        for line in stoplist_file:
            stoplist.append(line.strip())
    
    #1. choose the test data set and training kss
    test_set = me_set
    kss = fu_kss
    print('================ ¡ý Original ¡ý ================')
    output_result(test_set, kss)
    
    #2. use 'most_common_english_words' as stoplists and do not consider them 
    # Firstly, we create a new file 
    # to store the training dataset removed the stoplist words
    
    #================ ¡ý YOU CAN RERUN THIS PART ¡ý ================#
    with open(dataset_fu, 'r') as file:
        fu_set = file.readlines()  
    delete_set = delete_stoplist(fu_set, stoplist)

    if os.path.exists('.\delete_stoplist_full.txt'):
        os.remove('.\delete_stoplist_full.txt') # delete it at first
    
    with open('.\delete_stoplist_full.txt', 'w') as file:
        for item in delete_set:
            file.write(item + '\n')
    #================ ¡ü YOU CAN RERUN THIS PART ¡ü ================#  
    
    with open('.\delete_stoplist_full.txt', 'r') as file: # generate the new kss
        fu_kss_stoplst = extract_kss(file)
    # Notice that we do not need to tackle the training set
       
    # Then, we remove that the stoplist words in test set and re-train our model
    test_set_2 = delete_stoplist(test_set, stoplist)
    kss_2 = fu_kss_stoplst
    print('================ ¡ý delete Stoplist ¡ý ================')
    output_result(test_set_2, kss_2)    
    
    #3. explore the stemming
    
    #================ ¡ý YOU CAN RERUN THIS PART ¡ý ================#
    with open('.\delete_stoplist_full.txt', 'r') as file:
        fu_set = file.readlines()  
    stemm_set_p = tackle_stemmer(fu_set, 'p')
    stemm_set_l = tackle_stemmer(fu_set, 'l')
    stemm_set_s = tackle_stemmer(fu_set, 's')
    
    if os.path.exists('.\stemm_set_p_full.txt'):
        os.remove('.\stemm_set_p_full.txt') # delete it at first
    if os.path.exists('.\stemm_set_l_full.txt'):
        os.remove('.\stemm_set_l_full.txt') # delete it at first
    if os.path.exists('.\stemm_set_s_full.txt'):
        os.remove('.\stemm_set_s_full.txt') # delete it at first
    
    with open('.\stemm_set_p_full.txt', 'w') as file:
        for item in stemm_set_p:
            file.write(item + '\n')
    with open('.\stemm_set_l_full.txt', 'w') as file:
        for item in stemm_set_l:
            file.write(item + '\n')
    with open('.\stemm_set_s_full.txt', 'w') as file:
        for item in stemm_set_s:
            file.write(item + '\n')
    #================ ¡ü YOU CAN RERUN THIS PART ¡ü ================#  
    
    with open('.\stemm_set_p_full.txt', 'r') as file:
        fu_kss_stemm_p = extract_kss(file) 
    with open('.\stemm_set_l_full.txt', 'r') as file:
        fu_kss_stemm_l = extract_kss(file) 
    with open('.\stemm_set_s_full.txt', 'r') as file:
        fu_kss_stemm_s = extract_kss(file)
        
    print('================ ¡ý tackle Stemm ¡ý ================')
    
    print('>>>>>> PORTER <<<<<') # depend on the stoplist result
    test_set_3_1 = tackle_stemmer(test_set_2, 'p') 
    kss_3_1 = fu_kss_stemm_p
    output_result(test_set_3_1, kss_3_1) 
    
    print('>>>>>> LANCASTER <<<<<')
    test_set_3_2 = tackle_stemmer(test_set_2, 'l')
    kss_3_2 = fu_kss_stemm_l
    output_result(test_set_3_2, kss_3_2)  
    
    print('>>>>>> SNOWBALL <<<<<')
    test_set_3_3 = tackle_stemmer(test_set_2, 's')
    kss_3_3 = fu_kss_stemm_s    
    output_result(test_set_3_3, kss_3_3)      
    
    #4-1. explore the lemmatization
    
    #================ ¡ý YOU CAN RERUN THIS PART ¡ý ================#
    with open('.\delete_stoplist_full.txt', 'r') as file:
        fu_set = file.readlines()  
    lemma_set_n = tackle_lemma(fu_set, 'n')
    lemma_set_v = tackle_lemma(fu_set, 'v')
    
    if os.path.exists('.\lemma_set_n_full.txt'):
        os.remove('.\lemma_set_n_full.txt') # delete it at first
    if os.path.exists('.\lemma_set_v_full.txt'):
        os.remove('.\lemma_set_v_full.txt') # delete it at first
    
    with open('.\lemma_set_n_full.txt', 'w') as file:
        for item in lemma_set_n:
            file.write(item + '\n')

    with open('.\lemma_set_v_full.txt', 'w') as file:
        for item in lemma_set_v:
            file.write(item + '\n')
    #================ ¡ü YOU CAN RERUN THIS PART ¡ü ================#
    
    with open('.\lemma_set_n_full.txt', 'r') as file:
        fu_kss_lemma_n = extract_kss(file) 
    
    with open('.\lemma_set_v_full.txt', 'r') as file:
        fu_kss_lemma_v = extract_kss(file) 
        
    print('================ ¡ý tackle Lemma ¡ý ================')

    print('>>>>>> NOUN Lemma<<<<<') # none lemma
    kss_4_1 = fu_kss_lemma_n
    test_set_4_1 = tackle_lemma(test_set_2, 'n')
    output_result(test_set_4_1, kss_4_1)
    
    print('>>>>>> VERB Lemma<<<<<') # verb lemma
    kss_4_2 = fu_kss_lemma_v
    test_set_4_2 = tackle_lemma(test_set_2, 'v')
    output_result(test_set_4_2, kss_4_2)    
    
    #4-2. explore the lemmatization with POS Tag
    # In last part, I find we transfered (lemmatize) words without considering 
    # characteristic of words. The POS Tag can help we decide to use which
    # lematization method.
    
    #================ ¡ý YOU CAN RERUN THIS PART ¡ý ================#
    with open('.\delete_stoplist_full.txt', 'r') as file:
        fu_set = file.readlines()  
    lemma_set_pos_tag = tackle_lemma_2(fu_set)

    if os.path.exists('.\lemma_set_pos_tag_full.txt'):
        os.remove('.\lemma_set_pos_tag_full.txt') # delete it at first
    
    with open('.\lemma_set_pos_tag_full.txt', 'w') as file:
        for item in lemma_set_pos_tag:
            file.write(item + '\n')
    #================ ¡ü YOU CAN RERUN THIS PART ¡ü ================#
    with open('.\lemma_set_pos_tag_full.txt', 'r') as file:
        fu_kss_lemma_pos_tag = extract_kss(file) 
    
    print('================ ¡ý tackle Lemma with POS Tag ¡ý ================')
    kss_4_3 = fu_kss_lemma_pos_tag
    test_set_4_3 = tackle_lemma_2(test_set_2)
    output_result(test_set_4_3, kss_4_3)   