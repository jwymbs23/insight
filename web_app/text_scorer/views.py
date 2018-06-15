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
import pickle

@app.route('/input')
def text_input():
    return render_template("input.html")


@app.route('/output')
def text_output():
    #pull 'birth_month' from input field and store it
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
    pub_dict = {0: 'New York Times',1:'Breitbart', 2:'Washington Post'}

    
    #for i in pub_dict:
    plot_feature_comp(pub_dict, text_features, mean_features)
                                
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

    import base64
    figdata_png = base64.b64encode(figdata_png)

    
    return render_template("output.html", the_result = destination_paper, img_data=urllib.parse.quote(figdata_png))#urllib.parse.quote(canvas))#urllib.parse.quote(png_output))
