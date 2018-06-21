from sklearn.externals import joblib
from collections import Counter
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import string
import numpy as np
import pickle
import matplotlib.pyplot as plt

from text_scorer.pub_class import *
#from pub_class import *

#pub_dict = {0: 'New York Times',1:'Breitbart', 2:'Washington Post'}
pub_dict = {0: 'The Atlantic',1:'Breitbart', 2:'Buzzfeed News', 4:'Fox News', 5:'The Guardian', 6:'National Review', 7:'The New York Times',
            8:'Vox', 9:'The Washington Post'}


def model_t(fromUser  = 'Default', text = []):
    clf = joblib.load('./text_scorer/pickles/xg_clf_6_21.pkl')
    text_features = generate_features(text)
    print(text_features)
    destination = clf.predict([text_features])

    result = pub_dict[destination[0]]
    #if fromUser != 'Default':
    return result, text_features
    #else:
    #    return 'check your input'


def pre_process_text(text):
    #lower case text
    print(text)
    lower_case = text.lower()
    #replace strange quote characters with normal ones
    replace_quotes = text.replace('“', '"').replace('”', '"')
    #do punkt sentence tokenization
    sentence_tokenize = sent_detector.tokenize(text.strip())
    #do nltk word tokenization
    word_tokenize = nltk.word_tokenize(text)
    #get unique string counts
    string_counts = Counter(text)
    pos_tag = [nltk.pos_tag(text) for text in word_tokenize]
    return sentence_tokenize, word_tokenize, string_counts, pos_tag





def generate_features(text):
    sent_tok, word_tok, string_count, pos = pre_process_text(text)
    punctuation_set = set(string.punctuation)

    inst = Publication('', [sent_tok], [word_tok], [string_count], [pos])
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

#['word_count', 'sent_len', 'word_len', 'sent_len_std', 'unique_word_frac',
#'cps', 'qps', 'exps', 'foreign', 'flesch', 
#'RB_ps', 'RBR_ps', 'RBS_ps', 'WRB_ps', 'VB_ps',
#'VBD_ps', 'VBG_ps', 'VBN_ps', 'VBP_ps', 'VBZ_ps', 'JJ_ps', 'JJS_ps', 'JJR_ps',
#'said_ps', 'and_ps', 'but_ps', 'told_ps', 'i_ps', 'pronoun_ps', 'determiner_ps', 'preposition_ps', 'word_rarity']

    
    word_count = inst.word_count[0]
    sent_len = inst.sent_len[0]
    word_len = inst.word_len[0]
    sent_len_std = inst.sent_len_std[0]
    unique_word_frac = inst.unique_word_frac[0]
    cps = inst.cps[0]
    qps = inst.qps[0]
    exps = inst.exps[0]

    adverbs = inst.adverb_count[0]
    verbs = inst.verb_count[0]
    adjectives = inst.adj_count[0]

    foreign = inst.FW_count[0]
    print('-------------', foreign)
    
    flesch = inst.flesch_level[0]
    said_ps = inst.said_ps[0]
    and_ps = inst.and_ps[0]
    but_ps = inst.but_ps[0]
    told_ps = inst.told_ps[0]
    i_ps = inst.i_ps[0]
    pronoun = inst.pronoun_count[0]
    total_pronoun = inst.pronoun_count[0]
    total_determiner = inst.determiner_count[0]
    total_preposition = inst.prep_count[0]
    
    pronoun_ps = total_pronoun/inst.sent_count[0]
    determiner_ps = total_determiner/inst.sent_count[0]
    preposition_ps = total_preposition/inst.sent_count[0]
    
    
    
    inst.calc_word_rarity()
    word_rarity = inst.word_rarity[0]
                                        
    #flesch_five = inst.first_five
    #flesch_sec = inst.flesch_frac

    RB_ps  = inst.adverb_count[0]['RB']/inst.sent_count[0]#.apply(lambda row: row['RB'])/feature_df['inst.sent_count']
    RBR_ps = inst.adverb_count[0]['RBR']/inst.sent_count[0]
    RBS_ps = inst.adverb_count[0]['RBS']/inst.sent_count[0]
    WRB_ps = inst.adverb_count[0]['WRB']/inst.sent_count[0]
    
    #verb_subtypes
    VB_ps  = inst.verb_count[0]['VB' ]/inst.sent_count[0]
    VBD_ps = inst.verb_count[0]['VBD']/inst.sent_count[0]
    VBG_ps = inst.verb_count[0]['VBG']/inst.sent_count[0]
    VBN_ps = inst.verb_count[0]['VBN']/inst.sent_count[0]
    VBP_ps = inst.verb_count[0]['VBP']/inst.sent_count[0]
    VBZ_ps = inst.verb_count[0]['VBZ']/inst.sent_count[0]
    
    #adj_subtypes
    JJ_ps  = inst.adj_count[0]['JJ' ]/inst.sent_count[0]
    JJS_ps = inst.adj_count[0]['JJS']/inst.sent_count[0]
    JJR_ps = inst.adj_count[0]['JJR']/inst.sent_count[0]
    
    return [word_count, sent_len, word_len, sent_len_std, unique_word_frac,cps, qps, exps, foreign, flesch, 
            RB_ps, RBR_ps, RBS_ps, WRB_ps, VB_ps, VBD_ps, VBG_ps, VBN_ps, VBP_ps, VBZ_ps, JJ_ps, JJS_ps, JJR_ps,
            said_ps, and_ps, but_ps, told_ps, i_ps, pronoun_ps, determiner_ps, preposition_ps, word_rarity]



    
#    word_count = len([word for word in word_tok if word not in punctuation_set])
#    sent_count = len(sent_tok)
#    sent_len = float(word_count / sent_count)
#    sent_std = np.std([len(sent) for sent in sent_tok])#

#    unique_word_count = len(set([word for word in word_tok if word not in punctuation_set]))
#    unique_word_frac = float(unique_word_count / word_count)
#    mean_word_length = np.mean([len(word) for word in word_tok if word not in punctuation_set])

#    cps = string_counts[',']/sent_count
    #THIS ORDER MATTERS!!!!!!!!!! build order independence or awareness into the workflow?
    #wc (optional), sent_len, sent_len_std, unique_word_frac, word_len, cps
#    return [sent_len, sent_std, unique_word_frac, mean_word_length, cps]
    
def compare_to_mean(text_features, mean_features):
    return [[float(i/j) for i,j in zip(text_features, pub_features)] for pub_features in mean_features]


def plot_feature_comp(pub_dict, text_features, mean_features):
    features = ['word_count', 'sent_len', 'word_len', 'sent_len_std', 'unique_word_frac',
                'cps', 'qps', 'exps', 'foreign', 'flesch',
                'RB_ps', 'RBR_ps', 'RBS_ps', 'WRB_ps', 'VB_ps',
                'VBD_ps', 'VBG_ps', 'VBN_ps', 'VBP_ps', 'VBZ_ps', 'JJ_ps', 'JJS_ps', 'JJR_ps',
                'said_ps', 'and_ps', 'but_ps', 'told_ps', 'i_ps', 'pronoun_ps', 'determiner_ps', 'preposition_ps', 'word_rarity']
    
    plt.title('Features compared to average.')
    comp_to_mean = compare_to_mean(text_features, mean_features)
    
    fig, axs = plt.subplots(3, 3)
    
    colors = 'rgb'
    for i in pub_dict:
        axs[i].bar(list(range(5)), comp_to_mean[i][:5], color = colors[i])
        axs[i].bar(list(range(5)), [1 for _ in range(5)], color = 'black', alpha = 0.5)
        axs[i].set_title('{}'.format(pub_dict[i]))
        axs[i].set_xticks(range(5))
        axs[i].set_xticklabels(features, rotation = 45, ha = 'right')
    fig.subplots_adjust(bottom=0.3, left = 0.2, wspace = 0.3) #
    fig.set_size_inches(11, 11)
    
    #print(mean_features_comp[pub_id], pub_id)
    #plt.bar(list(range(5)), mean_features_comp, color = 'r')
    #plt.bar(list(range(5)), [1 for _ in range(5)], color = 'black', alpha = 0.5)
    #plt.ylim((0,1.2))
    #plt.xticks(range(5), features, rotation = 45, ha = 'right')
    plt.savefig('./text_scorer/tmp.png')#'{}.png'.format(pub_id))
