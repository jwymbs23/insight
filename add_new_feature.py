import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import string
import numpy as np
import random
from collections import Counter

import nltk
#nltk.download('averaged_perceptron_tagger')
#nltk.download('tagsets')
#nltk.download('cmudict')

#nltk.help.upenn_tagset()
#from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()
#import nltk.data
#sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

#from nltk.tag.stanford import StanfordPOSTagger

translator = str.maketrans('', '', string.punctuation + '”“')
punctuation_set = set(string.punctuation + '”“')


#arpabet = nltk.corpus.cmudict.dict()

from textstat.textstat import textstat


import glob


from pub_class import *

import operator


f_list = glob.glob('*_tokenized_strings.p')

for fname in f_list:
    pub_abbrev = fname[:-20]
    print(pub_abbrev)
    pub, article_id, auth, sent, word, string_count, pos = pickle.load(open(fname, 'rb'))
    
    inst = Publication(pub, sent, word, string_count, pos)
    #New feature to add

    
    #print(inst.adverb_count[0])
    feature_df = pickle.load(open(pub_abbrev + '_data_df.p', 'rb'))
    #new_feature_name


    #history of features that have been added this way (DO NOT DELETE):
    #inst.calc_word_count()
    #inst.calc_sent_count()
    #inst.calc_sent_len()
    #inst.calc_punc_ps()
#    feature_df['told_ps'] = inst.told_ps
    #feature_df['i_ps'] = inst.i_ps


    #inst.calc_sent_count()
    #inst.calc_pos_counts()

    #feature_df['total_pronoun'] = inst.pronoun_count
    #feature_df['total_determiner'] = inst.determiner_count
    #feature_df['total_preposition'] = inst.prep_count
    #
    #feature_df['pronoun_ps'] = feature_df['total_pronoun']/feature_df['sent_count']
    #feature_df['determiner_ps'] = feature_df['total_determiner']/feature_df['sent_count']
    #feature_df['preposition_ps'] = feature_df['total_preposition']/feature_df['sent_count']



    inst.calc_word_rarity()
    feature_df['word_rarity'] = inst.word_rarity



    #convert_pos = {'JJR': 'JJ', 'JJS': 'JJ', 'NNS': 'NN',
    #               'NNP': 'NN', 'NNPS': 'NN', 'RBR': 'RB',
    #               'RBS': 'RB', 'VBD': 'VB', 'VBG': 'VB',
    #               'VBN': 'VB', 'VBP': 'VB', 'VBZ': 'VB',
    #               '.': 'PP', ',': 'PP', "'": 'PP',
    #               '"': 'PP', ':':'PP', "''": 'PP', '(': 'PP', ')': 'PP', '``': 'PP'}
    #pos_string = []
    #for article_pos in pos:
    #    pos_string.append(' '.join([convert_pos[i[1]] if i[1] in convert_pos else i[1] for i in article_pos]))
    #feature_df['pos_string'] = pos_string
    #exit(0)
               

    

    #inst.calc_n_grams()
    #sorted_x = sorted(inst.gram_dict_pub_total.items(), key=operator.itemgetter(1))
    #print(sorted_x[::-1][:10])
    ##print(len(sorted_x))
    ##print(inst.gram_list[0][:10])
    ##print(inst.gram_list[1][:10])
    #for ci, i in enumerate(inst.gram_list):
    #    top_n = 20
    #    if len(i) < top_n:
    #        top_n = len(i)
    #    for j in range(top_n):
    #        #print(i[j][0])
    #        if i[j][0] not in list(feature_df):
    #            feature_df['ngram_'+i[j][0]] = [0 for k in range(len(feature_df))]
    #            feature_df.loc[ci, 'ngram_'+i[j][0]] = i[j][1]
    ##print(list(feature_df))
    #print(len(list(feature_df)))


#    exit(0)

    
    pickle.dump(feature_df, open(pub_abbrev+'_data_df.p', 'wb'))
