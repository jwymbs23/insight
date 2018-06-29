from flask import render_template
from flask import make_response
from text_scorer import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import pandas as pd
#import psycopg2
from flask import request
from text_scorer.model_t import model_t
from text_scorer.model_t import compare_to_mean
from text_scorer.model_t import plot_feature_comp
from text_scorer.model_t import plot_target_comp
from text_scorer.pub_class import *
import pickle
import base64
from sklearn.externals import joblib
import random
import urllib.parse
import datetime
from io import BytesIO
import matplotlib.pyplot as plt
    
id_to_pub = {0: 'The Atlantic',1:'Breitbart', 2:'Buzzfeed News', 3:'Fox News', 4:'The Guardian', 5:'National Review', 6:'The New York Times',
            7:'Vox', 8:'The Washington Post'}

pub_to_id = {'The Atlantic': 0,'Breitbart': 1, 'Buzzfeed News': 2, 'Fox News': 3, 'The Guardian': 4, 'National Review': 5, 'The New York Times': 6, 'Vox': 7, 'The Washington Post': 8}
#['atl', 'breit', 'buzz', 'fox', 'guard', 'natrev', 'nyt', 'vox', 'wapo']



#@app.route('/')
#def home_page():
#    return render_template("home.html")

@app.route('/slides')
def slides():
    return render_template("slides.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/')
def home_page():
    pub_names = sorted(list(id_to_pub.values()))
    return render_template("home.html", publications = pub_names)


@app.route('/output')
def text_output():
    clf = joblib.load('./text_scorer/pickles/stats_xgb_6_27.pkl')
    #pull 'birth_month' from input field and store it
    target_pub = request.args.get('publications')
    print('target_pub:', target_pub)
    text = request.args.get('text')
    #print(text)
    top_three_destinations, text_features = model_t(text = text, model = clf)
    top_destination = top_three_destinations[0][0]
    top_three = [{'name': i[0], 'score': '%d'%(int(i[1]*100))} for i in top_three_destinations]

    

    mean_features = pickle.load(open('./text_scorer/pickles/mean_features.p', 'rb'))
    print(mean_features)
    mean_features_comp = compare_to_mean(text_features, mean_features)
    #print(mean_features_comp)

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
    

    categories = {'Word Use': ['word_len', 'unique_word_frac', 'flesch', 'word_rarity', 'foreign','said_ps', 'told_ps'],
                  'Phrase Complexity': ['sent_len', 'sent_len_std' , 'cps', 'WRB_ps','and_ps', 'but_ps', 'VB_ps'],
                  'Parts of Speech': ['VBD_ps', 'VBG_ps', 'VBN_ps', 'VBP_ps', 'VBZ_ps', 'pronoun_ps', 'determiner_ps', 'preposition_ps'],
                  'Tone': ['qps', 'exps', 'RB_ps', 'RBS_ps', 'RBR_ps', 'JJ_ps', 'JJS_ps', 'JJR_ps' ,'i_ps']}
    

    
    
    #for i in pub_dict:
    #generate plot to compare input article to articles from the target publication across all features
    plot_target_comp(id_to_pub, pub_to_id, text_features, mean_features, target_pub, categories)


    figfile_target = BytesIO()
    plt.savefig(figfile_target, format='png')
    figfile_target.seek(0)  # rewind to beginning of file
    figdata_png_target = figfile_target.getvalue()


    figdata_png_target = base64.b64encode(figdata_png_target)

    

    #generate plot to compare input article to articles from all publications across top features
    plot_feature_comp(id_to_pub, pub_to_id, text_features, mean_features, target_pub)
    #from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    #from matplotlib.figure import Figure
    #from matplotlib.dates import DateFormatter
    #fig = Figure()
    #axis = fig.add_subplot(1, 1, 1)
    
    #xs = range(100)
    #ys = [random.randint(1, 50) for x in xs]
    
    #axis.plot(xs, ys)
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = figfile.getvalue()


    figdata_png = base64.b64encode(figdata_png)

    
    return render_template("output.html", top_three = top_three, the_target =  target_pub, img_data_target = urllib.parse.quote(figdata_png_target), img_data_total=urllib.parse.quote(figdata_png))#urllib.parse.quote(canvas))#urllib.parse.quote(png_output))
