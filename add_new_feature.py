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

    inst.calc_word_count()
    inst.calc_sent_count()
    inst.calc_sent_len()
    inst.calc_punc_ps()
    
    feature_df['told_ps'] = inst.told_ps
    
    pickle.dump(feature_df, open(pub_abbrev+'_data_df.p', 'wb'))
