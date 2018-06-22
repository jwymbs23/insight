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


id_to_pub = {0: 'The Atlantic',1:'Breitbart', 2:'Buzzfeed News', 3:'Fox News', 4:'The Guardian', 5:'National Review', 6:'The New York Times',
            7:'Vox', 8:'The Washington Post'}

pub_to_id = {'The Atlantic': 0,'Breitbart': 1, 'Buzzfeed News': 2, 'Fox News': 3, 'The Guardian': 4, 'National Review': 5, 'The New York Times': 6, 'Vox': 7, 'The Washington Post': 8}
#['atl', 'breit', 'buzz', 'fox', 'guard', 'natrev', 'nyt', 'vox', 'wapo']

@app.route('/input')
def text_input():
    pub_names = sorted(list(id_to_pub.values()))
    return render_template("input.html", publications = pub_names)


@app.route('/output')
def text_output():
    #pull 'birth_month' from input field and store it
    target_pub = request.args.get('publications')
    print('target_pub:', target_pub)
    text = request.args.get('text')
    #print(text)
    destination_paper, text_features = model_t(text = text)
    import urllib.parse
    import datetime
    from io import BytesIO
    import random

    import matplotlib.pyplot as plt
    mean_features = pickle.load(open('./text_scorer/pickles/mean_features.p', 'rb'))
    print(mean_features)
    mean_features_comp = compare_to_mean(text_features, mean_features)
    print(mean_features_comp)


    
    #for i in pub_dict:
    plot_target_comp(id_to_pub, pub_to_id, text_features, mean_features, target_pub)


    figfile_target = BytesIO()
    plt.savefig(figfile_target, format='png')
    figfile_target.seek(0)  # rewind to beginning of file
    figdata_png_target = figfile_target.getvalue()


    figdata_png_target = base64.b64encode(figdata_png_target)

    

    
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

    
    return render_template("output.html", the_result = destination_paper, the_target =  target_pub, img_data_target = urllib.parse.quote(figdata_png_target), img_data_total=urllib.parse.quote(figdata_png))#urllib.parse.quote(canvas))#urllib.parse.quote(png_output))
