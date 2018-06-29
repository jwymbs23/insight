
from collections import Counter
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import string
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from text_scorer.pub_class import *

from operator import itemgetter
#from pub_class import *

#pub_dict = {0: 'New York Times',1:'Breitbart', 2:'Washington Post'}
pub_dict = {0: 'The Atlantic',1:'Breitbart', 2:'Buzzfeed News', 3:'Fox News', 4:'The Guardian', 5:'National Review', 6:'The New York Times',
            7:'Vox', 8:'The Washington Post'}


def model_t(fromUser  = 'Default', text = [], model = joblib.load('./text_scorer/pickles/stats_xgb_6_27.pkl')):
    #clf = joblib.load('./text_scorer/pickles/xg_clf_6_21.pkl')
    text_features = generate_features(text)
    #print(text_features)
    destination = model.predict_proba([text_features])
    print(destination[0].argsort()[-3:][::-1])
    top_three_id = destination[0].argsort()[-3:][::-1]
    top_three = [(pub_dict[i], destination[0][i]) for i in top_three_id]#result = pub_dict[destination[0]]
    #if fromUser != 'Default':
    #result = top_three[0][0]
    return top_three, text_features
    #else:
    #    return 'check your input'


def pre_process_text(text):
    #lower case text
    print(text)
    lower_case = text.lower()
    #replace strange quote characters with normal ones
    replace_quotes = lower_case.replace('“', '"').replace('”', '"')
    #do punkt sentence tokenization
    sentence_tokenize = sent_detector.tokenize(lower_case.strip())
    #do nltk word tokenization
    word_tokenize = nltk.word_tokenize(lower_case)
    #get unique string counts
    string_counts = Counter(word_tokenize)
    pos_tag = nltk.pos_tag(word_tokenize)# for text in word_tokenize]
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
    
    return [sent_len, word_len, sent_len_std, unique_word_frac,cps, qps, exps, foreign, flesch, 
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



def plot_target_comp(id_to_pub, pub_to_id, text_features, mean_features, target_pub, categories):
    features = ['sent_len', 'word_len', 'sent_len_std', 'unique_word_frac',
                'cps', 'qps', 'exps', 'foreign', 'flesch',
                'RB_ps', 'RBR_ps', 'RBS_ps', 'WRB_ps', 'VB_ps',
                'VBD_ps', 'VBG_ps', 'VBN_ps', 'VBP_ps', 'VBZ_ps', 'JJ_ps', 'JJS_ps', 'JJR_ps',
                'said_ps', 'and_ps', 'but_ps', 'told_ps', 'i_ps', 'pronoun_ps', 'determiner_ps',
                'preposition_ps', 'word_rarity']
        

    human_readable_features = ['Sentence length', 'Word length', 'Sentence length variation', 'Unique word fraction',
                               'Commas /s', 'Questions /s', 'Exclamations /s', 'Foreign words',
                               'Flesch readability score', 'Adverbs /s', 'Comp. adverbs /s', 'Sup. adverbs /s',
                               'Wh-adverbs /s', 'Verbs /s', 'Past verbs /s', 'Gerunds /s', 'Past part. /s',
                               'Sing. present (not 3rd person) /s', 'Sing. present (3rd person) /s', 'Adjectives /s',
                               'Sup. adjectives /s', 'Comp. adjectives /s', 'Said /s', 'and /s', 'but /s', 'told /s',
                               'I /s', 'Pronouns /s', 'Determiners /s', 'Prepositions /s', 'Word rarity']
    
    
    comp_to_mean = compare_to_mean(text_features, mean_features)
    
    
    comp_to_target = {}
    for ci, i in enumerate(human_readable_features):
        comp_to_target[i] = comp_to_mean[pub_to_id[target_pub]][ci]
        
        
    fig,ax = plt.subplots(2,2)
    colors = ['#FDB927', '#006BB6', '#DA020E', '#00471B']
    for ci, category in enumerate(categories):
        #print(categories[category])
        #print(comp_to_target.items())
        sorted_comp_to_target = sorted( ((v,k) for k,v in comp_to_target.items() if features[human_readable_features.index(k)] in categories[category]), reverse=True)
        #print(sorted_comp_to_target)
        #print([i[0] for i in sorted_comp_to_target])
        ax[int(ci/2)%2, ci%2].set_title(category, fontsize=14)
        ax[int(ci/2)%2, ci%2].barh(range(len(categories[category])), [i[0] for i in sorted_comp_to_target if i[1] != 'Word count'], align='center',  color=colors[ci])
        ax[int(ci/2)%2, ci%2].barh(range(len(categories[category])), [1 for i in range(len(sorted_comp_to_target))], align='center',
                                   color='k', alpha = 0.5)
        ax[int(ci/2)%2, ci%2].set_yticks(range(len(categories[category])))
        ax[int(ci/2)%2, ci%2].set_yticklabels([i[1] for i in sorted_comp_to_target if i[1] != 'Word count'], fontsize=10)
        fig.subplots_adjust(left = 0.4) #
        ax[int(ci/2)%2, ci%2].invert_yaxis()  # labels read top-to-bottom
    fig.subplots_adjust(wspace = 1.5, hspace = 0.5) #
    fig.set_size_inches(10,5)
    fig.tight_layout()
    plt.savefig('./text_scorer/comp.png', bbox_inches = 'tight')
            





"""
    human_readable_features = ['Sentence length', 'Word length', 'Sentence length variation', 'Unique word fraction',
                               'Commas /s', 'Questions /s', 'Exclamations /s', 'Foreign words',
                               'Flesch readability score', 'Adverbs /s', 'Comp. adverbs /s', 'Sup. adverbs /s',
                               'Wh-adverbs /s', 'Verbs /s', 'Past verbs /s', 'Gerunds /s', 'Past part. /s',
                               'Sing. present (not 3rd person) /s', 'Sing. present (3rd person) /s', 'Adjectives /s',
                               'Sup. Adjectives /s', 'Comp. Adjectives /s', 'Said /s', 'and /s', 'but /s', 'told /s',
                               'I /s', 'Pronouns /s', 'Determiners /s', 'Prepositions /s', 'Word Rarity']

    
    comp_to_mean = compare_to_mean(text_features, mean_features)
    comp_to_target = {}
    for ci, i in enumerate(human_readable_features):
        comp_to_target[i] = comp_to_mean[pub_to_id[target_pub]][ci]

    fig,ax = plt.subplots()
    sorted_comp_to_target = sorted( ((v,k) for k,v in comp_to_target.items()), reverse=True)

    ax.barh([i for i in range(0,len(features))], [i[0] for i in sorted_comp_to_target if i[1] != 'Word count'], align='center',
                    color='r')
    ax.barh([i for i in range(0,len(features))], [1 for i in range(len(sorted_comp_to_target))], align='center',
                    color='k', alpha = 0.5)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([i[1] for i in sorted_comp_to_target if i[1] != 'Word count'])
    fig.subplots_adjust(left = 0.4) #
    ax.invert_yaxis()  # labels read top-to-bottom
    fig.set_size_inches(5,6)
    plt.savefig('./text_scorer/comp.png', bbox_inches = 'tight')
"""



def plot_feature_comp(id_to_pub, pub_to_id, text_features, mean_features, target_pub):
    features = [ 'sent_len', 'word_len', 'sent_len_std', 'unique_word_frac',
                'cps', 'qps', 'exps', 'foreign', 'flesch',
                'RB_ps', 'RBR_ps', 'RBS_ps', 'WRB_ps', 'VB_ps',
                'VBD_ps', 'VBG_ps', 'VBN_ps', 'VBP_ps', 'VBZ_ps', 'JJ_ps', 'JJS_ps', 'JJR_ps',
                'said_ps', 'and_ps', 'but_ps', 'told_ps', 'i_ps', 'pronoun_ps', 'determiner_ps', 'preposition_ps', 'word_rarity']
    human_readable_features = ['Sentence length', 'Word length', 'Sentence length variation', 'Unique word fraction',
                               'Commas /s', 'Questions /s', 'Exclamations /s', 'Foreign words',
                               'Flesch readability', 'Adverbs /s', 'Comparative adverbs /s', 'Superlative adverbs /s',
                               'Wh-adverbs /s', 'Verbs /s', 'Past tense verbs /s', 'Gerunds /s', 'Past participles /s',
                               'Singular present (not 3rd person) /s', 'Singular present (3rd person) /s', 'Adjectives /s',
                               'Superlative Adjectives /s', 'Comparative Adjectives /s', 'Said /s', 'and /s', 'but /s', 'told /s',
                               'I /s', 'Pronouns /s', 'Determiners /s', 'Prepositions /s', 'Word Rarity']
    
    plt.title('Features compared to average.')
    comp_to_mean = compare_to_mean(text_features, mean_features)

    fig, axs = plt.subplots(3, 3)
    ranked_most_important_features = ['Flesch readability', 'Unique word fraction', 'Said /s', 'Word length',
                                      'Commas /s', 'Foreign words']
    ranked_indices = [human_readable_features.index(i) for i in ranked_most_important_features]
    colors = ['#007DC3', '#F05133', '#241773', '#FDBB30', '#00843d', '#CC3433', '#4F8A10', '#8CCCE5', '#FB4F14']#'rgbcmyrgbcmyk'
    print(id_to_pub)
    for i in id_to_pub:
        print(i)
        print(comp_to_mean[i])
        axs[int(i/3)%3, i%3].bar(list(range(len(ranked_most_important_features))), [comp_to_mean[i][j] for j in ranked_indices], color = colors[i])
        axs[int(i/3)%3, i%3].bar(list(range(len(ranked_most_important_features))), [1 for _ in range(len(ranked_most_important_features))], color = 'black', alpha = 0.5)
        axs[int(i/3)%3, i%3].set_title('{}'.format(pub_dict[i]))
        axs[int(i/3)%3, i%3].set_xticks(range(len(ranked_most_important_features)))
        axs[int(i/3)%3, i%3].set_xticklabels(ranked_most_important_features, rotation = 45, ha = 'right')
    fig.subplots_adjust(bottom = 0.3, left = 0.2, wspace = 0.3, hspace = 0.8) #
    fig.set_size_inches(10, 10)
    fig.tight_layout()    
    #print(mean_features_comp[pub_id], pub_id)
    #plt.bar(list(range(5)), mean_features_comp, color = 'r')
    #plt.bar(list(range(5)), [1 for _ in range(5)], color = 'black', alpha = 0.5)
    #plt.ylim((0,1.2))
    #plt.xticks(range(5), features, rotation = 45, ha = 'right')
    plt.savefig('./text_scorer/tmp.png')#'{}.png'.format(pub_id))
