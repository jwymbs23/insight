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
    inst.calc_word_count()
    inst.calc_sent_count()
    inst.calc_sent_len()
    #inst.calc_hook_first_five()
    #inst.calc_hook_frac()
    inst.calc_unique_words()
    inst.calc_word_length()
    inst.calc_sent_len_std()
    inst.calc_flesch_level()
    inst.calc_pos_counts()
    inst.calc_punc_ps()
    

    
    #print(inst.adverb_count[0])
    
    feature_df = pd.DataFrame(
        {'label': [pub for i in range(len(inst.word_count))],
         'word_count': inst.word_count,
         'sent_len': inst.sent_len,
         'word_len': inst.word_len,
         'sent_len_std': inst.sent_len_std,
         'unique_word_frac': inst.unique_word_frac,
         'cps': inst.cps,
         'qps': inst.qps,
         'exps': inst.exps,
         'adverbs': inst.adverb_count,
         'verbs': inst.verb_count,
         'adjectives' : inst.adj_count,
         'foreign' : inst.FW_count,
         'flesch': inst.flesch_level,
         'said_ps': inst.said_ps,
         'and_ps': inst.and_ps,
         'but_ps': inst.but_ps,
         'flesch_five': inst.first_five,
         'flesch_sec': inst.flesch_frac 
         #'sent_compound': [vec[0] for vec in inst.sentiment_vec]
        })


    #totals
    feature_df['total_adv'] = feature_df['adverbs'].apply(lambda row: sum(row.values()))
    feature_df['total_verb'] = feature_df['verbs'].apply(lambda row: sum(row.values()))
    feature_df['total_adj'] = feature_df['adjectives'].apply(lambda row: sum(row.values()))
    #sentence_count
    feature_df['sent_count'] = feature_df['word_count']/feature_df['sent_len']
    
    #per_sentence
    feature_df['adv_ps'] = feature_df['total_adv']/(feature_df['sent_count'])
    feature_df['verb_ps'] = feature_df['total_verb']/(feature_df['sent_count'])
    feature_df['adj_ps'] = feature_df['total_adj']/(feature_df['sent_count'])
    ##################
    #VB, VBD, VBG, VBN, VBP, VBZ
    #JJ, JJS, JJR
    #RB, RBS, RBR, WRB
    #################
    #adverb_subtypes
    feature_df['RB_ps'] = feature_df['adverbs'].apply(lambda row: row['RB'])/feature_df['sent_count']
    feature_df['RBR_ps'] = feature_df['adverbs'].apply(lambda row: row['RBR'])/feature_df['sent_count']
    feature_df['RBS_ps'] = feature_df['adverbs'].apply(lambda row: row['RBS'])/feature_df['sent_count']
    feature_df['WRB_ps'] = feature_df['adverbs'].apply(lambda row: row['WRB'])/feature_df['sent_count']
    
    #verb_subtypes
    feature_df['VB_ps'] = feature_df['verbs'].apply(lambda row: row['VB'])/feature_df['sent_count']
    feature_df['VBD_ps'] = feature_df['verbs'].apply(lambda row: row['VBD'])/feature_df['sent_count']
    feature_df['VBG_ps'] = feature_df['verbs'].apply(lambda row: row['VBG'])/feature_df['sent_count']
    feature_df['VBN_ps'] = feature_df['verbs'].apply(lambda row: row['VBN'])/feature_df['sent_count']
    feature_df['VBP_ps'] = feature_df['verbs'].apply(lambda row: row['VBP'])/feature_df['sent_count']
    feature_df['VBZ_ps'] = feature_df['verbs'].apply(lambda row: row['VBZ'])/feature_df['sent_count']
    
    #adj_subtypes
    feature_df['JJ_ps'] = feature_df['adjectives'].apply(lambda row: row['JJ'])/feature_df['sent_count']
    feature_df['JJS_ps'] = feature_df['adjectives'].apply(lambda row: row['JJS'])/feature_df['sent_count']
    feature_df['JJR_ps'] = feature_df['adjectives'].apply(lambda row: row['JJR'])/feature_df['sent_count']
    
    pickle.dump(feature_df, open(pub_abbrev+'_data_df.p', 'wb'))
    
    
#
#pub_list_labels = ['guardian']
#for i in [inst]:
#    plt.hist(i.word_count, bins = list(range(0,3000,50)), normed = True, alpha = 0.7)
#    plt.legend(labels = pub_list_labels)
#    plt.title("Word count")
#    plt.show()




#[pub_nyt, nyt_feature_df, pub_breit, breit_feature_df,
#              pub_wapo, wapo_feature_df, pub_guard, guard_feature_df] =  pickle.load(open( "nyt_data_df.p", "rb" ))


