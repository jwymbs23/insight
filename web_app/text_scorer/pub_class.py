import string
import numpy as np
import textstat as ts
from textstat.textstat import textstat

from collections import Counter

translator = str.maketrans('', '', string.punctuation + '”“')
punctuation_set = set(string.punctuation + '”“')



class Publication():

        def __init__(self, pub_id = '', pub_sent = [], pub_word = [], pub_string_count = [], pub_pos = []):
            self.publication_name = pub_id
            #self.id_list = pub_id_list
            self.sent_tok = pub_sent
            self.word_tok = pub_word
            self.string_count = pub_string_count
            self.pos = pub_pos
            
        def calc_word_count(self):
            self.word_count_punc = [len(text) for text in self.word_tok]
            self.word_count = [len([word for word in text if word not in punctuation_set]) for text in self.word_tok]
            
        def calc_sent_count(self):
            self.sent_count = [len(text) for text in self.sent_tok]
            
        def calc_sent_len(self):
            if not self.word_count:
                self.calc_word_count()
            if not self.sent_count:
                self.calc_sent_count
            self.sent_len = [float(i/j) for i,j in zip(self.word_count, self.sent_count)]
            
        def calc_unique_words(self):
            self.unique_wc = [len(set([word for word in text if word not in punctuation_set])) for text in self.word_tok]
            self.unique_word_frac = [float(i/j) for i,j in zip(self.unique_wc, self.word_count)]
            
        def calc_word_length(self):
            self.word_len = [np.mean([len(word) for word in text if word not in punctuation_set]) for text in self.word_tok]
            
        def calc_sent_len_std(self):
            self.sent_len_std = [np.std([len(sent) for sent in text]) for text in self.sent_tok]



        def calc_punc_ps(self):
            if not self.sent_count:
                self.calc_sent_count()
            print(self.string_count, self.sent_count)
            self.cps = [float(counter[',']/sent_num) for counter, sent_num in zip(self.string_count, self.sent_count)]
            self.qps = [float(counter['?']/sent_num) for counter, sent_num in zip(self.string_count, self.sent_count)]
            self.exps = [float(counter['!']/sent_num) for counter, sent_num in zip(self.string_count, self.sent_count)]
            self.said_ps = [float(counter['said']/sent_num) for counter, sent_num in zip(self.string_count, self.sent_count)]
            self.but_ps = [float(counter['but']/sent_num) for counter, sent_num in zip(self.string_count, self.sent_count)]
            self.and_ps = [float(counter['and']/sent_num) for counter, sent_num in zip(self.string_count, self.sent_count)]
            self.told_ps = [float(counter['told']/sent_num) for counter, sent_num in zip(self.string_count, self.sent_count)]
            self.i_ps = [float(counter['i']/sent_num) for counter, sent_num in zip(self.string_count, self.sent_count)]
            
        def calc_pos_counts(self):
            #tag text:
            self.adverb_count = []
            self.verb_count = []
            self.adj_count = []
            self.FW_count = []
            self.pronoun_count = []
            self.prep_count = []
            self.determiner_count = []
            for ci, article in enumerate(self.pos):
                if not ci%1000:
                    print(ci)
                article_adverb_dict = {'RB': 0, 'RBR': 0, 'RBS': 0, 'WRB': 0}
                article_verb_dict = {'VB': 0, 'VBD': 0, 'VBG': 0, 'VBN': 0 , 'VBZ': 0, 'VBP': 0}
                article_adj_dict = {'JJ': 0, 'JJR': 0, 'JJS': 0}
                article_FW = 0
                article_prep = 0
                article_pronoun = 0
                article_determiner = 0
                print('[[[[[[[', article)
                for word in article:
                        if 'RB' in word[1]:
                                #print(word[0], word[1])
                                #RB: adverb
                                #RBR: adverb, comparative
                                #RBS: adverb, superlative
                                #WRB: what, where, who, when
                                article_adverb_dict[word[1]] += 1
                                #article_adv_count += 1
                        if 'VB' in word[1]:
                                #VB: verb, base form
                                #VBD: verb, past tense
                                #VBG: verb, present participle or gerund
                                #VBN: verb, past participle
                                #VBP: verb, present tense, not 3rd person singular
                                #VBZ: verb, present tense, 3rd person singular
                                article_verb_dict[word[1]] += 1
                                #article_verb_count += 1
                        if 'JJ' in word[1]:
                                #JJ: adjective or numeral, ordinal
                                #JJR: adjective, comparative
                                #JJS: adjective, superlative
                                article_adj_dict[word[1]] += 1
                        if 'FW' == word[1]:
                                #foreign word
                                article_FW += 1
                        if 'IN' == word[1]:
                                article_prep += 1
                        if 'PRP' in word[1]:
                                article_pronoun += 1
                        if 'DT' == word[1]:
                                article_determiner += 1
                self.adverb_count.append(article_adverb_dict)
                self.verb_count.append(article_verb_dict)
                self.adj_count.append(article_adj_dict)
                self.FW_count.append(article_FW)
                self.prep_count.append(article_prep)
                self.pronoun_count.append(article_pronoun)
                self.determiner_count.append(article_determiner)


        def calc_word_rarity(self):
                word_freq_dict = {}
                with open('count_1w.txt') as wc_f:
                        lines = wc_f.readlines()
                        for ci,line in enumerate(lines):
                                word, count = line.strip().split()
                                word_freq_dict[word] = ci#float(1./int(count))
                self.word_rarity = [np.mean([10000 if word not in word_freq_dict else word_freq_dict[word]
                                             for word in text if word not in punctuation_set]) for text in self.word_tok]


                
        def calc_sentiment_polarity(self):
            self.sentiment_vec = []
            for ci, article in enumerate(self.sent_tok):
                if not ci%1000:
                    print(ci)
                sentence_count = len(article)
                #compound, positive, neutral, negative
                document_score = [0,0,0,0]
                for sentence in article:
                    #print(sentence)
                    ss = sid.polarity_scores(sentence)
                    document_score[0] += ss['compound']
                    document_score[1] += ss['pos']
                    document_score[2] += ss['neu']
                    document_score[3] += ss['neg']
                    #print(ss)
                    #for k in sorted(ss):
                    #    print('{0}: {1}, '.format(k, ss[k]), end='')
                    #    print()
                    #input()
                document_score = [float(i/sentence_count) for i in document_score]
            self.sentiment_vec.append(document_score)
        
        def calc_flesch_level(self):
                #206.835 - (total words/total sentences) * 1.015 - (total syllables / total words) * 84.6
                #print(sum([len(word)/3. if word not in arpabet else len(arpabet[word][0]) for word in self.word_tok[0] if word not in punctuation_set]))
                #print(self.sent_len[0], self.word_count[0])
                #input()
                #self.flesch_level = [sl * 0.39 + 11.8 * np.mean(
                #    [len(word)/3. if word not in arpabet else len(arpabet[word][0])
                #     for word in text if word not in punctuation_set]) - 15.59
                #                     for text, sl, wc in zip(self.word_tok, self.sent_len, self.word_count)]
                self.flesch_level = [textstat.flesch_reading_ease(' '.join(text)) for text in self.sent_tok]
        

        def calc_hook_first_five(self):
                self.first_five = [textstat.flesch_reading_ease(' '.join(text[:5])) if len(text) > 20 else 0 for text in self.sent_tok]

        def calc_hook_frac(self):
                n_sections = 4
                frac = 1./n_sections
                self.flesch_frac = []
                for text in self.sent_tok:
                        n_sentences_per_split = int(np.floor(len(text)*frac))
                        if n_sentences_per_split > 5:
                                section_score = []
                                for sec in range(n_sections):
                                        start_point = n_sentences_per_split*sec
                                        end_point = n_sentences_per_split*(sec+1) if sec < n_sections else len(text)
                                        flesch_section = textstat.flesch_reading_ease(' '.join(text[start_point: end_point]))
                                        section_score.append(flesch_section)
                                self.flesch_frac.append(section_score)
                        else:
                                self.flesch_frac.append([0 for i in range(n_sections)])           


        def calc_n_grams(self):
                grams = 2
                converg_pos = {'JJR': 'JJ', 'JJS': 'JJ', 'NNS': 'NN',
                               'NNP': 'NN', 'NNPS': 'NN', 'RBR': 'RB',
                               'RBS': 'RB', 'VBD': 'VB', 'VBG': 'VB',
                               'VBN': 'VB', 'VBP': 'VB', 'VBZ': 'VB',
                               '.': 'PP', ',': 'PP', "'": 'PP',
                               '"': 'PP', ':':'PP', "''": 'PP', '(': 'PP', ')': 'PP', '``': 'PP'}
                self.gram_list = []
                self.gram_dict_pub_total = {}
                for text in self.pos:
                        num_ngrams = len(text) - grams
                        norm_increment = 1./num_ngrams
                        gram_sequence = []
                        for token in range(len(text) - grams):
                                sequence = ' '.join([converg_pos[text[i][1]] if text[i][1] in converg_pos else text[i][1] for i in range(token, token + grams)])
                                gram_sequence.append(sequence)
                                if sequence not in self.gram_dict_pub_total:
                                        self.gram_dict_pub_total[sequence] = norm_increment
                                else:
                                        self.gram_dict_pub_total[sequence] += norm_increment
                                #print(gram_sequence[-1])
                        #self.gram_list.append(Counter(gram_sequence))
                        c = Counter(gram_sequence)
                        #print(c.most_common(5))
                        self.gram_list.append(c.most_common(10))#print(len(gram_sequence), len(Counter(gram_sequence)))
