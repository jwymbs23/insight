from sklearn.externals import joblib
from collections import Counter
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import string
import numpy as np
import pickle
import matplotlib.pyplot as plt

pub_dict = {0: 'New York Times',1:'Breitbart', 2:'Washington Post'}

def model_t(fromUser  = 'Default', text = []):
    clf = joblib.load('./text_scorer/pickles/decision_tree_10.pkl')
    text_features = generate_features(text)
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
    return sentence_tokenize, word_tokenize, string_counts


    
def generate_features(text):
    sent_tok, word_tok, string_counts = pre_process_text(text)
    punctuation_set = set(string.punctuation)
    word_count = len([word for word in word_tok if word not in punctuation_set])
    sent_count = len(sent_tok)
    sent_len = float(word_count / sent_count)
    sent_std = np.std([len(sent) for sent in sent_tok])

    unique_word_count = len(set([word for word in word_tok if word not in punctuation_set]))
    unique_word_frac = float(unique_word_count / word_count)
    mean_word_length = np.mean([len(word) for word in word_tok if word not in punctuation_set])

    cps = string_counts[',']/sent_count
    #THIS ORDER MATTERS!!!!!!!!!! build order independence or awareness into the workflow?
    #wc (optional), sent_len, sent_len_std, unique_word_frac, word_len, cps
    return [sent_len, sent_std, unique_word_frac, mean_word_length, cps]
    
def compare_to_mean(text_features, mean_features):
    return [[float(i/j) for i,j in zip(text_features, pub_features)] for pub_features in mean_features]


def plot_feature_comp(pub_dict, text_features, mean_features):
    features = ['sent length', 'sent variation', 'unique word frac', 'word length', 'commas per sent']
    plt.title('Features compared to average.')
    comp_to_mean = compare_to_mean(text_features, mean_features)
    
    fig, axs = plt.subplots(1, 3)
    
    colors = 'rgb'
    for i in pub_dict:
        axs[i].bar(list(range(5)), comp_to_mean[i], color = colors[i])
        axs[i].bar(list(range(5)), [1 for _ in range(5)], color = 'black', alpha = 0.5)
        axs[i].set_title('{}'.format(pub_dict[i]))
        axs[i].set_xticks(range(5))
        axs[i].set_xticklabels(features, rotation = 45, ha = 'right')
    fig.subplots_adjust(bottom=0.3, left = 0.2, wspace = 0.3) #
    fig.set_size_inches(11, 3.8)
    
    #print(mean_features_comp[pub_id], pub_id)
    #plt.bar(list(range(5)), mean_features_comp, color = 'r')
    #plt.bar(list(range(5)), [1 for _ in range(5)], color = 'black', alpha = 0.5)
    #plt.ylim((0,1.2))
    #plt.xticks(range(5), features, rotation = 45, ha = 'right')
    plt.savefig('./text_scorer/tmp.png')#'{}.png'.format(pub_id))
