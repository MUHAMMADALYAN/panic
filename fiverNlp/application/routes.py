from flask import Flask, render_template, request, url_for, flash,  send_from_directory, send_file, jsonify
#from flask import current_app as app
from werkzeug.utils import secure_filename, redirect
import json
import inspect, nltk
import numpy as np

#new imports for Bert
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import csv, pickle
from tqdm import tqdm

from forms import (
    FileInputForm,
    PredictionDataForm,
    TrainModelForm,
    ChangeClassColorsForm,
    #SubmitAllForm,
    special_form
)

from util_functions import *

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.math as tfmath
import tensorflow.keras.backend as tfbackend

from os import mkdir
import sys
import requests
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
import re
from flask_executor import Executor
import _thread
from timeloop import Timeloop
import datetime
from pytz import timezone
import urllib.parse

from sentence_transformers import SentenceTransformer, util

import webhoseio

webhoseio.config(token="8018e387-9258-4fd4-9ec5-9f9366a779a8")

app = Flask(__name__)
app.config.from_object('config.Config')
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://mikenyc:12345@localhost/mike'
# app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#COUNTRY_CODES = ['','US','AU','HK','GB']
#MARKET_LANGUAGE_CODES = ['','english','spanish','french']

COUNTRY_CODES = ["","US","AF","AL","DZ","AS","AD","AO","AI","AQ","AG","AR","AM","AW","AU","AT","AZ","BS","BH","BD","BB","BY","BE","BZ","BJ","BM","BT","BO","BA","BW","BR","IO","VG","BN","BG","BF","BI","KH","CM","CA","CV","KY","CF","TD","CL","CN","CX","CC","CO","KM","CK","CR","HR","CU","CW","CY","CZ","CD","DK","DJ","DM","DO","TL","EC","EG","SV","GQ","ER","EE","ET","FK","FO","FJ","FI","FR","PF","GA","GM","GE","DE","GH","GI","GR","GL","GD","GU","GT","GG","GN","GW","GY","HT","HN","HK","HU","IS","IN","ID","IR","IQ","IE","IM","IL","IT","CI","JM","JP","JE","JO","KZ","KE","KI","XK","KW","KG","LA","LV","LB","LS","LR","LY","LI","LT","LU","MO","MK","MG","MW","MY","MV","ML","MT","MH","MR","MU","YT","MX","FM","MD","MC","MN","ME","MS","MA","MZ","MM","NA","NR","NP","NL","AN","NC","NZ","NI","NE","NG","NU","KP","MP","NO","OM","PK","PW","PS","PA","PG","PY","PE","PH","PN","PL","PT","PR","QA","CG","RE","RO","RU","RW","BL","SH","KN","LC","MF","PM","VC","WS","SM","ST","SA","SN","RS","SC","SL","SG","SX","SK","SI","SB","SO","ZA","KR","SS","ES","LK","SD","SR","SJ","SZ","SE","CH","SY","TW","TJ","TZ","TH","TG","TK","TO","TT","TN","TR","TM","TC","TV","VI","UG","UA","AE","GB","UY","UZ","VU","VA","VE","VN","WF","EH","YE","ZM","ZW"]
MARKET_LANGUAGE_CODES = ["","english","afrikaans","albanian","amharic","arabic","armenian","azerbaijani","basque","belarusian","bengali","bulgarian","burmese","catalan","cherokee","chinese","chineset","croatian","czech","danish","dhivehi","dutch","estonian","finnish","french","galician","georgian","german","greek","gujarati","hebrew","hindi","hungarian","icelandic","ignore","indonesian","inuktitut","irish","italian","japanese","kannada","khmer","korean","laothian","latvian","lithuanian","macedonian","malay","malayalam","maltese","norwegian","oriya","persian","polish","portuguese","punjabi","romanian","russian","serbian","sinhalese","slovak","slovenian","spanish","swahili","swedish","syriac","tagalog","tamil","telugu","thai","tibetan","turkish","ukrainian","urdu","vietnamese","welsh","yiddish"]


SITE_TYPES = ['','news','blogs','discussions']
REST = ['Day','Week','Month']

db = SQLAlchemy(app)
from models import *
db.create_all()
db.session.commit()
executor = Executor(app)

tl = Timeloop()
tz = timezone('EST')

GET  = 'GET'
POST = 'POST'
TEST_STRING = ''

tokenizer 	 = load_tokenizer()
model     	 = load_model('static/Models/model_under_use.h5')
maxlen    	 = 40
class_colors = load_classColors()
#print(1111111111111111111,class_colors,file=sys.stderr)
sentence_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')



@tl.job(interval=datetime.timedelta(minutes=300))
def day():
    work('Day')
    report_word('Daily')

    # reports = ReportModel.query.filter_by(frequency='Daily',up_to_date=True).all()
    # if(len(reports)>0):
    #     for i in reports:
    #         report_background(i.id,i.type,i.first,i.second,i.range_from,i.range_to)
@tl.job(interval=datetime.timedelta(days=7))
def week():
    work('Week')
    report_word('Weekly')
    # reports = ReportModel.query.filter_by(frequency='Weekly',up_to_date=True).all()
    # if(len(reports)>0):
    #     for i in reports:
    #         report_background(i.id,i.type,i.first,i.second,i.range_from,i.range_to)
@tl.job(interval=datetime.timedelta(days=30))
def month():
    work('Month')
    report_word('Monthly')
    # reports = ReportModel.query.filter_by(frequency='Monthly',up_to_date=True).all()
    # if(len(reports)>0):
    #     for i in reports:
    #         report_background(i.id,i.type,i.first,i.second,i.range_from,i.range_to)


def work(fetch_frequency):
    searchqueries = SearchQueryModel.query.filter_by(fetch_frequency=fetch_frequency,status='playing').all()
    if(searchqueries==[]):
        print('Empty',file=sys.stderr)
        return None

    for searchquery in searchqueries:
        try:
            print(searchquery.title,file=sys.stderr)
            temp = SearchQueryModel(title=searchquery.title,query_string=searchquery.query_string,market_language_code=searchquery.market_language_code,country_code=searchquery.country_code,site=searchquery.site,site_type=searchquery.site_type,characters=searchquery.characters,freshness=2,fetch_frequency=searchquery.fetch_frequency)
            search_query_documents_background(temp)
            db.session.commit()
            db.session.expunge_all()
            db.session.close()
        except Exception as e:
            print('error 123',e)
def report_word(frequency):
    reports = ReportModel.query.filter_by(frequency=frequency,up_to_date=True).all()
    if(len(reports)>0):
        for i in reports:
            default = False
            if('default: ' in i.title):
                default = True
            report_background(i.id,i.type,i.first,i.second,i.range_from,i.range_to,default)



@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.expunge_all()
    db.session.remove()




@app.route("/",  methods=[GET])
@app.route("/home",  methods=[GET])
def home():
    return render_template('home.html')

@app.route("/changeClassColors",  methods=[GET, POST])
def change_class_colors():
	global class_colors

	form = ChangeClassColorsForm()
	if form.validate_on_submit():
		new_purpose		 = form.new_purpose.data
		new_craftsmaship = form.new_craftsmaship.data
		new_aesthetic 	 = form.new_aesthetic.data
		new_narrative	 = form.new_narrative.data

		save_classColors(new_purpose, new_craftsmaship, new_aesthetic, new_narrative)
		class_colors = load_classColors()

		flash("Class Colors Changed Successfully!", "success")
		return redirect(url_for("home"))

	return render_template("changeClassColors.html",  form=form, class_colors=class_colors)

@app.route("/train", methods=[GET, POST])
def train():
    inputform  = FileInputForm()

    if inputform.validate_on_submit():

        file = inputform.file.data
        if file.filename.split(".")[-1] != 'tsv':

            flash("ONLY UPLOAD A 'tsv' FILE!", "danger")
            return redirect(url_for('train'))

        singlefile(file)
        flash("File Successfully Uploaded", "success")
        file.close()

    
    trainModelform   = TrainModelForm()        

    return render_template("train.html", inputform=inputform, trainModelform=trainModelform)

@app.route('/train_model/<retrain>', methods=[GET, POST])
def train_model(retrain):
    global model, tokenizer, maxlen

    if retrain == 'True':
        file_path       = "bin/output2.tsv"
    else:
        file_path = "static/File_Upload_Folder/uploaded.tsv"

    try:
        #data     = np.genfromtxt(file_path, delimiter='\t', dtype= str, encoding="utf8")
        data=loadTSVfromFolder()
        data=np.array(data)
       
        features   = data[:, 0]
       
        labels = data[:, 1]
        print("These are the model labels: ", labels)       #TODO: remove print
        

        if len(features) < 70:
            flash("There must be atleast 70 rows of Data before training", "danger")
            return "less than 70"


        total_samples = data.shape[0]

        #Cleaning stop words and converting to lists
        features = filter_func(features)

        #shuffling the data    
        features, labels = shuffle(features, labels)

        #Imp numbers to create Embeddings and for padding
        #maxlen, count = count_words(features)
        #num_words     = len(count)
        #maxlen        = maxlen 

        #One hot encoding Labels
        labels = onehot_encode_labels(labels)

        #Tokenizing the data
        #tokenizer.fit_on_texts(features)
	
	tok_features=tokenize(features,tokenizer)
        #print("tok_features ------> ",tok_features)

        input_ids_in=(tok_features[0])
        input_masks_in=(tok_features[1])
        print("input_ids_in.shape ---> ",input_ids_in.shape)
        print("input_masks_in.shape --> ",input_masks_in.shape)

        #Getting Embeddings
        cls_token=embeddings(input_ids_in,input_masks_in)
        print("cls_token.shape ---> ",cls_token.shape)
        total_samples = len(cls_token)

        embedding_features = np.asarray(cls_token)


        #Training the Model
        model.fit(embedding_features, labels, epochs=50)

        # saving the tokenizer
        #save_tokenizer(tokenizer)

        #feature_sequences = tokenizer.texts_to_sequences(features)
        #feature_padded = pad_sequences(feature_sequences, maxlen=maxlen, padding='post', truncating='post')

        #Training the Model
        #model.fit(feature_padded, labels, epochs=20)

        #overwriting the model
        model.save('static/Models/model_under_use.h5')

        flash("Model Trained and Saved!", "success")
        return "training done"

    except ValueError as ve:
        flash("ERROR, Plz check if all your sentences end with a period i.e ' . '",  "danger")
        print(ve)
        return "ValueError"

    except KeyError as ke:
        flash("ERROR, Encountered an Unknown Label during Training, Please Check Training Data",  "danger")
        print(ke)
        return "keyError"

    except OSError as oe:
        flash("ERROR! No File uploded to Train on", "danger")
        print(oe)
        return "osError"

    except IndexError as ie:
        flash("There must be atleast 70 rows of Data before training", "danger")
        return "indexError"

@app.route("/restrat_model", methods=[POST])
def restart_model():
	global model

	model = load_model("static/Models/scratch_model.h5")
	model.save("static/Models/model_under_use.h5")

	flash("Model Started Form Scratch Successful", "success")

	return redirect(url_for('train'))


@app.route("/test",  methods=[GET, POST])
def test():
    global TEST_STRING

    predictionForm = PredictionDataForm()

    if predictionForm.validate_on_submit():
            
           TEST_STRING = predictionForm.text_area.data
           return redirect(url_for("results"))
    
    return render_template("test.html", predictionForm=predictionForm)


@app.route("/results", methods=[GET, POST])
def results():

    global model, tokenizer, maxlen, TEST_STRING  

    sentences   = nltk.sent_tokenize(TEST_STRING)

    #text_seq        = tokenizer.texts_to_sequences(sentences)
    #text_seq_padded = pad_sequences(text_seq, maxlen=maxlen, padding='post', truncating='post')
		
    tok_test_features=tokenize(sentences,tokenizer)

    test_input_ids_in=(tok_test_features[0])
    test_input_masks_in=(tok_test_features[1])
        

    #Getting Embeddings
    test_cls_token=embeddings(test_input_ids_in,test_input_masks_in)
    embedding_features = np.asarray(test_cls_token)


    predictions = model.predict(embedding_features)
    #print("predictions===",predictions)    
	
    #predictions = model.predict(text_seq_padded)
        
    class_num = tfmath.argmax(predictions, axis= 1)
    class_num = tfbackend.eval(class_num)
    labels    = decode_onehot_labels(class_num)

    specialForm = special_form(labels)
    selects     = [
        getattr(specialForm, f"special_{i}")
        for i in range(specialForm.n_attrs)
    ]
    
    data = list(zip(sentences, roundoff(predictions), labels, selects))
    bin_data = loadTSVfromBin()
    print("\n\nBIN LEN:", len(bin_data), '\n\n')

    if specialForm.validate_on_submit():
        corrected_labels = [
            sel.data
            for sel in selects
        ]

        appendTSVtoBin(corrected_labels, sentences)

        flash(f"Added { len(corrected_labels) } rows to the bin, Now total rows in bin are { len(bin_data)+len(corrected_labels) }", "success")
        return redirect(url_for("proceed"))
        

    return render_template(
        "results.html",
        data            = data,
        len_data        = len(data),
        bin_data        = bin_data,
        len_bin_data    = len(bin_data),
        class_colors    = class_colors,
        specialForm     = specialForm
    )


@app.route("/download_file", methods=[GET])
def download_file():
    path = "bin/output2.tsv"
    return send_file(path, as_attachment=True)

@app.route("/clear_bin", methods=[GET])
def clear_bin():

    clearBin()
    flash("Bin Emptied", "success")
    return redirect(url_for("home"))


@app.route("/proceed", methods=[GET])
def proceed():
    return render_template("proceed.html")


@app.route("/predict", methods=["GET","POST"])
def classified():
    text=request.json
    text=text['mytext']
    global model, tokenizer , maxlen
    sentences   = nltk.sent_tokenize(text) 
    tok_test_features=tokenize(sentences,tokenizer)

    test_input_ids_in=(tok_test_features[0])
    test_input_masks_in=(tok_test_features[1])
        

    #Getting Embeddings
    test_cls_token=embeddings(test_input_ids_in,test_input_masks_in)
    embedding_features = np.asarray(test_cls_token)


    predictions = model.predict(embedding_features)
    #print("predictions===",predictions)    
	
    #text_seq        = tokenizer.texts_to_sequences(sentences)
    #text_seq_padded = pad_sequences(text_seq, maxlen=maxlen,padding='post', truncating='post')
    #predictions = model.predict(text_seq_padded)
    class_num = tfmath.argmax(predictions, axis= 1) #Returns the index with the largest value across axes of a tensor.
    class_num = tfbackend.eval(class_num) 
    labels    = decode_onehot_labels(class_num)

    dict={

    }
    dat = zip(sentences, labels)
    for i in dat:
        dict[i[0]]=i[1]
    return json.dumps(dict)

@app.route("/getsimilarity",methods=['POST'])
def sentenceSimliarity():
    dimensions = {'aesthetic':0,'craftsmanship':0,'purpose':0,'narrative':0}
    data = request.get_json(force=True)
    sentence1 = data.get('sentence1')
    sentence2 = data.get('sentence2')
    a = getSimlarity(sentence1,sentence2)
    return jsonify(a)
def getSimlarity(sentence1,sentence2):
    s1 = []
    s2 = []
    for i in sentence1:
        s1.append(i[0])
    for i in sentence2:
        s2.append(i[0])
    embeddings1 = sentence_model.encode(s1, convert_to_tensor=True)
    embeddings2 = sentence_model.encode(s2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    #Output the pairs with their score
    d = {}
    for i in range(len(sentence1)):
      # s = []
      dd = {}
      for j in range(len(sentence2)):
        # print(sentence1[i],sentence2[j])
        # s.append("{:.2f}".format(cosine_scores[i][j]))
        dd[sentence2[j][0]] = {'similarity':"{:.2f}".format(cosine_scores[i][j]),'title':sentence2[j][1],'type':sentence2[j][2]}
      print(dd)
      d[sentence1[i]] = dd
    return d


@app.route("/addToBin",methods=["GET","POST"])
def addToBin():
    text=request.json
    lab=text['labels']
    sen=text['sentences']

    x = zip(sen,lab)
    json_data_tuples = list(x)
    
    bin_data=loadTSVfromBin()
    
    for i in json_data_tuples:
            for j in bin_data:
                #comparing only sentences
                if i[0]==j[0]:
                    bin_data.remove(j)

    
    bin_data_labels=[]
    bin_data_sentences=[]
    for i in bin_data:
            bin_data_sentences.append(i[0])
            bin_data_labels.append(i[1])
   
    writeTSVtoBin(bin_data_labels,bin_data_sentences)
    appendTSVtoBin(lab,sen)
    return ("bin_data")

@app.route("/seeBin", methods=[GET,POST])
def seeBin():

    bin_data = loadTSVfromBin()
    bin_labels=[]
    bin_sentences=[]
    for i in bin_data:
        bin_labels.append(i[1])
        bin_sentences.append(i[0])
    

    specialForm = special_form(bin_labels)
    selects     = [
        getattr(specialForm, f"special_{i}")
        for i in range(specialForm.n_attrs)
    ]

    if specialForm.validate_on_submit():
        corrected_labels = [
            sel.data
            for sel in selects
        ]

        writeTSVtoBin(corrected_labels, bin_sentences) 

    data=list(zip(bin_sentences,bin_labels,selects))    
    threshold = Threshold.query.filter_by(id=1).first()
    if(threshold is None):
        threshold = Threshold(id=1,value=100)
        db.session.add(threshold)
        db.session.commit()
    return render_template(
        "seeBin.html",
        data=data,
        bin_data        = bin_data,
        len_bin_data    = len(bin_data),
        class_colors    = class_colors,
        specialForm     = specialForm,
        threshold       = threshold.value
    )






@app.route('/threshold')
def threshold():
    threshold = Threshold.query.filter_by(id=1).first()
    if(threshold):
        return render_template('threshold.html',threshold=threshold.value)
    else:
        return render_template('threshold.html',threshold=100)

@app.route('/setthreshold')
def set_threshold():
    threshold = Threshold.query.filter_by(id=1).first()
    t = request.args.get('t')
    if(t is None or t==''):
        return 'error'
    if(threshold is None):
        threshold = Threshold(id=1,value=t)
        db.session.add(threshold)
        db.session.commit()
    else:
        threshold.value = t
        db.session.commit()
    return redirect(url_for('seeBin'))

@app.route('/<type>/savetobin',methods=['POST'])
def save_to_bin(type):
    dimensions = ["aesthetic","narrative","craftsmanship","purpose"]
    title = request.form.get('title')
    sentence = request.form.get('sentence')
    if(title is None or title=='' or title=='None'):
        return 'error'
    temp = None
    if(type=='companies'):
        temp = CompanyDocumentModel.query.filter_by(title=title).first()
    elif(type=='searchquerydocuments'):
        temp = SearchQueryDocumentModel.query.filter_by(title=title).first()
        if(temp is None):
            temp = SearchQueryDocumentModel.query.filter(SearchQueryDocumentModel.f_title==title).filter(SearchQueryDocumentModel.classified_sentences.contains(sentence)).first()
    elif(type=='arbitrarydocuments'):
        temp = ArbitraryDocumentModel.query.filter_by(title=title).first()
    else:
        return 'error 0'
    if(temp and temp.classified_sentences):
        sentences = eval(temp.classified_sentences)
        url = url_for('addToBin',_external=True)
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        labels_data = []
        sentences_data = []
        for i in sentences:
            if(sentences[i] in dimensions):
                labels_data.append(sentences[i])
                sentences_data.append(i)
        if(len(labels_data)==len(sentences_data)):
            myobj = {'labels': labels_data,'sentences': sentences_data}
            myobj=json.dumps(myobj)
            x = requests.post(url, data = myobj,headers=headers)
            if(x.ok):
                return 'done'
            else:
                return 'error 1'
        else:
            return 'error 2'
    else:
        return 'error 3'



@app.route('/<type>/saveclassifier',methods=['POST'])
def save_classifier(type):
    dimensions = ["aesthetic","narrative","craftsmanship","purpose"]
    title = request.form.get('title')
    dimension = request.form.get('dimension')
    sentence = request.form.get('sentence')
    if(title is None or title=='' or sentence is None or sentence=='' or dimension is None or dimension not in dimensions):
        return 'error'
    temp = None
    if(type=='companies'):
        temp = CompanyDocumentModel.query.filter_by(title=title).first()
    elif(type=='searchquerydocuments'):
        temp = SearchQueryDocumentModel.query.filter_by(title=title).first()
        if(temp is None):
            temp = SearchQueryDocumentModel.query.filter(SearchQueryDocumentModel.f_title==title).filter(SearchQueryDocumentModel.classified_sentences.contains(sentence)).first()
    elif(type=='arbitrarydocuments'):
        temp = ArbitraryDocumentModel.query.filter_by(title=title).first()
    else:
        return 'error'
    if(temp):
        if(temp and temp.classified_sentences):
            sentences = eval(temp.classified_sentences)
            if(sentences.get(sentence)):
                sentences[sentence] = dimension
                temp.classified_sentences = str(sentences)
        else:
            return 'cannot find company'
        db.session.commit()
    return 'done'




@app.route('/<type>/classifier',methods=['POST'])
def classifier(type):
    sentences = None
    title = request.form.get('title')
    query_document_title = request.form.get('query_document_title')
    highlight_sentence = request.form.get('sentence')
    if(query_document_title is None):
        if(title is None or title==''):
            return 'title is empty'
    else:
        if(query_document_title is None or query_document_title==''):
            return 'query_document_title is empty'
        title = query_document_title
    if(type=='companies'):
        temp = CompanyDocumentModel.query.filter_by(title=title).first()
        if(temp and temp.classified_sentences):
            sentences = eval(temp.classified_sentences)
        else:
            return 'cannot find company'
    elif(type=='searchquerydocuments'):
        print(query_document_title,title,highlight_sentence)
        if(query_document_title):
            temp = SearchQueryDocumentModel.query.filter_by(title=query_document_title).first()
        else:
            temp = SearchQueryDocumentModel.query.filter(SearchQueryDocumentModel.f_title==title).filter(SearchQueryDocumentModel.classified_sentences.contains(highlight_sentence)).first()
        print(temp)
        if(temp and temp.classified_sentences):
            sentences = eval(temp.classified_sentences)
        else:
            return 'error'
    elif(type=='arbitrarydocuments'):
        temp = ArbitraryDocumentModel.query.filter_by(title=title).first()
        if(temp and temp.classified_sentences):
            sentences = eval(temp.classified_sentences)
        else:
            return 'error'
    elif(type=='tags'):
        temp = None
        for i in CompanyDocumentModel.query.all():
            if(i.industry_tags and title in i.industry_tags and i.classified_sentences and highlight_sentence in i.classified_sentences):
                temp = i
                break
        if(temp is None):
            for i in ArbitraryDocumentModel.query.all():
                if(i.industry_tags and title in i.industry_tags and i.classified_sentences and highlight_sentence in i.classified_sentences):
                    temp = i
                    break
        if(temp and temp.classified_sentences):
            sentences = eval(temp.classified_sentences)
        else:
            return 'error'
    else:
        return 'error 1'
    dimensions = ["aesthetic","narrative","craftsmanship","purpose"]
    #print(highlight_sentence,file=sys.stderr)
    return render_template('classifier.html',sentences=sentences,dimensions=dimensions,title=title,highlight_sentence=highlight_sentence,class_colors=class_colors)



#@app.route('/')
#def home():
#    return render_template('home.html')



@app.route('/industrytags/')
def industry_tags_route():
    try:
        industry_tags = IndustryTags.query.all()
        return render_template('industrytags.html',industry_tags=industry_tags)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'

@app.route('/industrytags/add',methods=['POST'])
def industry_tags_add():
    try:
        result = request.form
        company_industry_tags = result.get('company_industry_tags')
        arbitrary_industry_tags = result.get('arbitrary_industry_tags')
        if(company_industry_tags):
            old_title = result.get('old_title')
            if(old_title is None or old_title==''):
                return 'error'
            companydocument = CompanyDocumentModel.query.filter_by(title=old_title).first()
            if(companydocument):
                if(companydocument.industry_tags is None or companydocument.industry_tags==''):
                    companydocument.industry_tags = company_industry_tags
                else:
                    if(company_industry_tags in companydocument.industry_tags):
                        return 'industry tag already exists in this company'
                    companydocument.industry_tags += ','+company_industry_tags
                db.session.commit()
                return redirect(url_for('edit_company')+'?title='+old_title)
        elif(arbitrary_industry_tags):
            old_title = result.get('old_title')
            if(old_title is None or old_title==''):
                return 'error'
            arbitrarydocument = ArbitraryDocumentModel.query.filter_by(title=old_title).first()
            if(arbitrarydocument):
                if(arbitrarydocument.industry_tags is None or arbitrarydocument.industry_tags==''):
                    arbitrarydocument.industry_tags = arbitrary_industry_tags
                else:
                    if(arbitrary_industry_tags in arbitrarydocument.industry_tags):
                        return 'industry tag already exists in this company'
                    arbitrarydocument.industry_tags += ','+arbitrary_industry_tags
                db.session.commit()
                return redirect(url_for('edit_arbitrary_document')+'?title='+old_title)

        else:
            title = result.get('title')
            if(title and title!=''):
                industry_tag = IndustryTags(title=title)
            else:
                return 'empty title'
            db.session.add(industry_tag)
            db.session.commit()
            return redirect(url_for('industry_tags_route'))
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'


@app.route('/industrytags/edit',methods=['POST','GET'])
def industry_tags_edit():
    if(request.method=='POST'):
        # try:
            result = request.form
            print(result,file=sys.stderr)
            title = result.get('title')
            old_title = result.get('old_title')

            if(old_title and old_title!=''):
                industry_tag = IndustryTags.query.filter_by(title=old_title).first()
            else:
                return 'empty title 0'
            if(industry_tag):
                if(not title or title==''):
                    return 'empty title 1'
                industry_tag.title = title
                db.session.commit()
                companydocuments = CompanyDocumentModel.query.all()
                print(companydocuments,type(companydocuments),file=sys.stderr)
                arbitrarydocuments = ArbitraryDocumentModel.query.all()
                if(len(companydocuments)>0):
                    for companydocument in companydocuments:
                        if(companydocument.industry_tags and old_title in companydocument.industry_tags):
                            companydocument.industry_tags = companydocument.industry_tags.replace(old_title,title)
                        db.session.commit()
                if(len(arbitrarydocuments)>0):
                    for arbitrarydocument in arbitrarydocuments:
                        if(arbitrarydocument.industry_tags and old_title in arbitrarydocument.industry_tags):
                            arbitrarydocument.industry_tags = arbitrarydocument.industry_tags.replace(old_title,title)
                        db.session.commit()
            else:
                return 'empty title 2'
            return redirect(url_for('industry_tags_route'))
        # except Exception as e:
        #     print(e,file=sys.stderr)
        #     return 'error'
    else:
        try:
            title = request.args.get('title')
            if(title and title!=''):
                industry_tag = IndustryTags.query.filter_by(title=title).first()
                if(not industry_tag):
                    return 'error'
            else:
                return 'error'
            return render_template('editindustrytag.html',industry_tag=industry_tag)
        except Exception as e:
            print(e,file=sys.stderr)
            return 'error'

@app.route('/industrytags/delete',methods=['POST'])
def industry_tags_delete():
    try:
        result = request.form
        company_industry_tags = result.get('company_industry_tags')
        arbitrary_industry_tags = result.get('arbitrary_industry_tags')
        if(company_industry_tags):
            old_title = result.get('old_title')
            title = result.get('title')
            if(old_title is None or old_title==''):
                return 'error'
            companydocument = CompanyDocumentModel.query.filter_by(title=old_title).first()
            if(companydocument):
                list_tags = companydocument.industry_tags.split(',')
                print(company_industry_tags,file=sys.stderr)
                list_tags.remove(company_industry_tags)
                list_tags = ','.join(list_tags)
                if(list_tags==''):
                    list_tags = None
                companydocument.industry_tags = list_tags
                db.session.commit()
                return redirect(url_for('edit_company')+'?title='+old_title)
        elif(arbitrary_industry_tags):
            old_title = result.get('old_title')
            title = result.get('title')
            if(old_title is None or old_title==''):
                return 'error'
            arbitrarydocument = ArbitraryDocumentModel.query.filter_by(title=old_title).first()
            if(arbitrarydocument):
                list_tags = arbitrarydocument.industry_tags.split(',')
                print(arbitrary_industry_tags,file=sys.stderr)
                list_tags.remove(arbitrary_industry_tags)
                list_tags = ','.join(list_tags)
                if(list_tags==''):
                    list_tags = None
                arbitrarydocument.industry_tags = list_tags
                db.session.commit()
            return redirect(url_for('edit_arbitrary_document')+'?title='+old_title)
        else:
            title = result.get('title')
            if(title=='' or title is None):
                return 'no search query document found'
            if(IndustryTags.delete(title=title)):
                companydocuments = CompanyDocumentModel.query.all()
                arbitrarydocuments = ArbitraryDocumentModel.query.all()
                if(len(companydocuments)>0):
                    for companydocument in companydocuments:
                        if(companydocument.industry_tags and title in companydocument.industry_tags):
                            list_tags = companydocument.industry_tags.split(',')
                            list_tags.remove(title)
                            list_tags = ','.join(list_tags)
                            if(list_tags==''):
                                list_tags = None
                            companydocument.industry_tags = list_tags
                            db.session.commit()
                if(len(arbitrarydocuments)>0):
                    for arbitrarydocument in arbitrarydocuments:
                        if(arbitrarydocument.industry_tags and title in arbitrarydocument.industry_tags):
                            list_tags = arbitrarydocument.industry_tags.split(',')
                            list_tags.remove(title)
                            list_tags = ','.join(list_tags)
                            if(list_tags==''):
                                list_tags = None
                            arbitrarydocument.industry_tags = list_tags
                            db.session.commit()
                    return redirect(url_for('industry_tags_route'))
            else:
                return 'error'
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'








@app.route('/searchqueries/deletesearchquerydocument',methods=['POST'])
def delete_search_query_document():
    result = request.form
    title = result.get('title')
    print(title,file=sys.stderr)
    if(title=='' or title is None):
        return 'no search query document found'
    try:
        f_title = SearchQueryDocumentModel.query.filter_by(title = title).first().f_title
        if(SearchQueryDocumentModel.delete(title=title)):
            return redirect(url_for('search_query_documents',title=f_title))
        else:
            return 'error'
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'


@app.route('/searchqueries/seeclassifiedsentences',methods=['POST'])
def search_queries_classified_sentences():
    result = request.form
    title = result.get('title')
    if(title is None or title==''):
        return "title cannot be empty"
    try:
        searchquerydocument = SearchQueryDocumentModel.query.filter_by(title = title).first()
        if(searchquerydocument is None):
            return 'Error'
        d = eval(searchquerydocument.classified_sentences)
        return render_template('seeclassifiedsentences.html',classified_sentences=searchquerydocument.classified_sentences)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'

@app.route('/searchqueries/savesearchquerydocument',methods=['POST'])
def save_search_query_document():
    try:
        result = request.form
        clean_text = result.get('clean_text')
        if(clean_text):
            clean_text = re.sub(r"(\r\n|\r|\n)","",clean_text)
            clean_text = re.sub(r"([^0-9]\.)",r"\1 ",clean_text)
        searchquerydocument = SearchQueryDocumentModel(title=result.get('title'),clean_text=clean_text,date=result.get('date'),author=result.get('author'),provider=result.get('provider'),url=result.get('url'),image_url=result.get('image_url'))
        old_title = result.get('old_title')
        if(searchquerydocument.title is None or searchquerydocument.title==''):
            return "title cannot be empty"
        if(old_title is None or old_title==''):
            return "Error"
        if(searchquerydocument.clean_text=='' or searchquerydocument.clean_text=='None'):
            searchquerydocument.clean_text = None
        if(searchquerydocument.clean_text is not None):
            # searchquerydocument.classified_sentences = str(temp_azure(searchquerydocument.clean_text))
            res = requests.post(url_for('classified',_external=True), json={"mytext":re.sub(r'[^a-zA-Z0-9. ]','',searchquerydocument.clean_text)})
            if res.ok:
                searchquerydocument.classified_sentences = str(res.json())
            else:
                searchquerydocument.classified_sentences = None
        else:
            searchquerydocument.classified_sentences = None
        SearchQueryDocumentModel.query.filter_by(title = old_title).update(dict(title=searchquerydocument.title,clean_text=searchquerydocument.clean_text,classified_sentences=str(searchquerydocument.classified_sentences),date=searchquerydocument.date,author=searchquerydocument.author,provider=searchquerydocument.provider,url=searchquerydocument.url,image_url=searchquerydocument.image_url))
        db.session.commit()
        return redirect(url_for('search_query_documents',title=SearchQueryDocumentModel.query.filter_by(title = searchquerydocument.title).first().f_title))
    except Exception as e:
        print(e,file=sys.stderr)
        return "Error"

@app.route('/searchqueries/changestatus',methods=['POST'])
def search_query_change_status():
    title = request.form.get('title')
    print(title,file=sys.stderr)
    if(title is None or title==''):
        return 'error'
    searchquery = SearchQueryModel.query.filter_by(title = title).first()
    print(title,file=sys.stderr)
    if(searchquery.status=='playing'):
        searchquery.status = 'paused'
    elif(searchquery.status=='paused'):
        searchquery.status = 'playing'
    else:
        return 'error'
    try:
        db.session.commit()
    except:
        return 'error'
    return 'done'


@app.route('/searchqueries/editsearchquerydocument',methods=["POST"])
def edit_search_query_document():
    title = request.form.get('title')
    if(title=='' or title is None):
        return 'no search query document found'
    searchquerydocument = SearchQueryDocumentModel.query.filter_by(title = title).first()
    if(searchquerydocument is not None):
        return render_template('editsearchquerydocument.html',searchquerydocument=searchquerydocument)
    else:
        return 'Error'

@app.route('/searchqueries/searchquerydocuments')
def search_query_documents():
    data = ''
    title = request.args.get('title')
    if(title=='' or title is None):
        return 'no search query document found'
    sort = request.args.get('sort')
    searchquerydocuments = []
    for i in SearchQueryDocumentModel.query.filter_by(f_title = title).all():
        searchquerydocuments.append(i)
    if(sort=='titleup'):
        searchquerydocuments.sort(key=lambda x: x.title.lower(), reverse=True)
    elif(sort=='titledown'):
        searchquerydocuments.sort(key=lambda x: x.title.lower(), reverse=False)
    elif(sort=='sourceup'):
        searchquerydocuments.sort(key=lambda x: x.provider.lower(), reverse=True)
    elif(sort=='sourcedown'):
        searchquerydocuments.sort(key=lambda x: x.provider.lower(), reverse=False)
    elif(sort=='dateup'):
        searchquerydocuments.sort(key=lambda x: x.date, reverse=True)
    elif(sort=='datedown'):
        searchquerydocuments.sort(key=lambda x: x.date, reverse=False)
    elif(sort == 'processingup'):
        temp = []
        for i in searchquerydocuments:
            if(i.clean_text and len(i.clean_text)>500):
                temp.append([i,True])
            else:
                temp.append([i,False])
        temp.sort(key=lambda x: x[1], reverse=True)
        searchquerydocuments = []
        for i in temp:
            searchquerydocuments.append(i[0])
    elif(sort == 'processingdown'):
        temp = []
        for i in searchquerydocuments:
            if(i.clean_text and len(i.clean_text)>500):
                temp.append([i,True])
            else:
                temp.append([i,False])
        temp.sort(key=lambda x: x[1], reverse=False)
        searchquerydocuments = []
        for i in temp:
            searchquerydocuments.append(i[0])

    return render_template('searchquerydocuments.html',searchquerydocuments=searchquerydocuments,title=title)



@app.route('/search',methods=['POST']) 
def search():
    search = request.form.get('q')
    title = request.form.get('title')
    # search = 'example first lowercase article'
    search = search.lower()
    search = search.split(' ')
    query = [] 
    search = [k for k in search if k != '']
    clauses = [ SearchQueryDocumentModel.title.like('%{0}%'.format(k)) for k in search]
    temp_articles = SearchQueryDocumentModel.query.filter(or_(*clauses)).all()
    articles = []
    for i in temp_articles:
        t = []
        heading = i.title.lower().split(' ')
        heading = [k for k in heading if k != '']
        flag = False
        for s in heading:
            for q in search:
                if (s == q):
                    flag = True
                    # print(s,q,file=sys.stderr)
        if (flag):
            t.append(i)
            t.append(0)
            for s in heading:
                for q in search:
                    if (s == q):
                        t[1] += 1
            articles.append(t)
    articles.sort(key=lambda x: x[1], reverse=True)
    result = []
    for i in articles:
        if(i[0].f_title==title):
            result.append(i[0])
    return render_template('searchquerydocuments.html',searchquerydocuments=result,title=title)






@app.route('/searchqueries/')
def search_queries():
    try:
        searchqueries = SearchQueryModel.get_rows()
        total = []
        #new
        yesterday = datetime.datetime.now(tz) - datetime.timedelta(days = 1)
        new_documents = []
        for i in searchqueries:
            total.append(len(SearchQueryDocumentModel.query.filter_by(f_title=i.title).all()))
            #new
            new_documents.append(len(SearchQueryDocumentModel.query.filter(SearchQueryDocumentModel.f_title==i.title).filter(SearchQueryDocumentModel.date_created > yesterday).all()))
    except Exception as e:
        print(e,file=sys.stderr)
    #new
    return render_template('searchqueries.html',searchqueries=searchqueries,total=total,new_documents=new_documents,zip=zip)

@app.route('/searchqueries/addnewsearchquery')
def add_new_search_query():
    return render_template('addeditsearchqueries.html',COUNTRY_CODES=COUNTRY_CODES,MARKET_LANGUAGE_CODES=MARKET_LANGUAGE_CODES,SITE_TYPES=SITE_TYPES,REST=REST,searchquery=SearchQueryModel(title='',query_string='',characters='',site=''))

@app.route('/searchqueries/editsearchquery')
def edit_search_queries():
    title = request.args.get('title')
    if(title=='' or title is None):
        return 'no search query found'
    try:
        searchquery = SearchQueryModel.get_row_by_title(title)
        return render_template('addeditsearchqueries.html',COUNTRY_CODES=COUNTRY_CODES,MARKET_LANGUAGE_CODES=MARKET_LANGUAGE_CODES,SITE_TYPES=SITE_TYPES,REST=REST,searchquery=searchquery)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'

@app.route('/searchqueries/deletesearchquery')
def delete_search_query():
    title = request.args.get('title')
    if(title=='' or title is None):
        return 'no search query found'
    try:
        if(SearchQueryModel.delete(title=title) and SearchQueryDocumentModel.deleteall(title=title)):
            return redirect(url_for('search_queries'))
        else:
            return 'error'
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'


@app.route('/searchqueries/savesearchquery',methods=['POST'])
def savesearchquery():
    try:
        result = request.form
        old_title = result.get('old_title')
        title = result.get('title')
        query_string = result.get('query_string')
        market_language_code = result.get('market_language_code')
        country_code = result.get('country_code')
        site_type = result.get('site_type')
        site = result.get('site')
        characters = result.get('characters')
        freshness = 30
        fetch_frequency = result.get('fetch_frequency')
        if(title is None or title==''):
            return "title cannot be empty"
        if(query_string is None or query_string==''):
            return "query string cannot be empty"
        if(site=='' or site==None):
            site = ''
        if(characters=='' or characters==None):
            characters = ''
        searchquery = SearchQueryModel(title=title,query_string=query_string,market_language_code=market_language_code,country_code=country_code,site_type=site_type,site=site,freshness=freshness,characters=characters,fetch_frequency=fetch_frequency,status='playing')
        if(old_title is not None and old_title!=''):
            SearchQueryModel.query.filter_by(title = old_title).update(dict(title=title,query_string=query_string,market_language_code=market_language_code,country_code=country_code,site_type=site_type,characters=characters,freshness=freshness,fetch_frequency=fetch_frequency,site=site))
        else:
            db.session.add(searchquery)
        temp = SearchQueryModel(title=searchquery.title,query_string=searchquery.query_string,market_language_code=searchquery.market_language_code,country_code=searchquery.country_code,site=searchquery.site,site_type=searchquery.site_type,characters=characters,freshness=searchquery.freshness,fetch_frequency=searchquery.fetch_frequency)
        executor.submit(search_query_documents_background,temp)
        db.session.commit()
        db.session.expunge_all()
        db.session.close()

        return redirect(url_for('search_queries'))

    except Exception as e:
        print(e,file=sys.stderr)
        return "Error"




@app.route('/arbitrarydocuments/')
def arbitrary_documents():
    try:
        documents = ArbitraryDocumentModel.query.all()
    except Exception as e:
        print(e,file=sys.stderr)
    return render_template('arbitrarydocuments.html',documents=documents)

@app.route('/arbitrarydocuments/addnew')
def add_new_document():
    try:
        arbitrarydocument = ArbitraryDocumentModel(title='',author='',provider='',url='',image_url='',date='',industry_tags=None,clean_text='',classified_sentences='')
        industrytags = IndustryTags.query.all()
        return render_template('addeditarbitrarydocument.html',arbitrarydocument=arbitrarydocument,industrytags=industrytags)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'

@app.route('/arbitrarydocuments/edit')
def edit_arbitrary_document():
    result = request.args
    title = result.get('title')
    title = urllib.parse.unquote(title)
    print(title,file=sys.stderr)
    if(title=='' or title is None):
        return 'no arbitrary document found'
    try:
        arbitrarydocument = ArbitraryDocumentModel.query.filter_by(title=title).first()
        industrytags = IndustryTags.query.all()
        return render_template('addeditarbitrarydocument.html',arbitrarydocument=arbitrarydocument,industrytags=industrytags)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'


@app.route('/arbitrarydocuments/savearbitrarydocument',methods=['POST'])
def save_arbitrary_document():
    try:
        result = request.form
        clean_text = result.get('clean_text')
        if(clean_text):
            clean_text = re.sub(r"(\r\n|\r|\n)","",clean_text)
            clean_text = re.sub(r"([^0-9]\.)",r"\1 ",clean_text)
        arbitrarydocument = ArbitraryDocumentModel(title=result.get('title'),clean_text=clean_text,date=result.get('date'),author=result.get('author'),provider=result.get('provider'),url=result.get('url'),image_url=result.get('image_url'),industry_tags=result.get('industry_tags'))
        old_title = result.get('old_title')
        if(arbitrarydocument.title is None or arbitrarydocument.title==''):
            return "title cannot be empty"
        if(arbitrarydocument.clean_text=='' or arbitrarydocument.clean_text=='None'):
            arbitrarydocument.clean_text = None
        if(arbitrarydocument.clean_text is not None):
            # arbitrarydocument.classified_sentences = str(temp_azure(arbitrarydocument.clean_text))
            res = requests.post(url_for('classified',_external=True), json={"mytext":re.sub(r'[^a-zA-Z0-9. ]','',arbitrarydocument.clean_text)})
            if res.ok:
                arbitrarydocument.classified_sentences = str(res.json())
            else:
                arbitrarydocument.classified_sentences = None
        else:
            arbitrarydocument.classified_sentences = None
        if(old_title is not None and old_title!=''):
            ArbitraryDocumentModel.query.filter_by(title = old_title).update(dict(title=arbitrarydocument.title,clean_text=arbitrarydocument.clean_text,classified_sentences=str(arbitrarydocument.classified_sentences),date=arbitrarydocument.date,author=arbitrarydocument.author,provider=arbitrarydocument.provider,url=arbitrarydocument.url,image_url=arbitrarydocument.image_url,industry_tags=arbitrarydocument.industry_tags))
            db.session.commit()
        else:
            db.session.add(arbitrarydocument)
            db.session.commit()
        return redirect(url_for('arbitrary_documents'))
    except Exception as e:
        db.session.rollback()
        print(e,file=sys.stderr)
        return "Error"

@app.route('/arbitrarydocuments/delete')
def delete_arbitrary_document():
    title = request.args.get('title')
    if(title=='' or title is None):
        return 'no arbitrary document found'
    try:
        if(ArbitraryDocumentModel.delete(title=title)):
            return redirect(url_for('arbitrary_documents'))
        else:
            return 'delete error'
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'

@app.route('/arbitrarydocuments/seeclassifiedsentences')
def see_classified_sentences_arbitrary():
    title = request.args.get('title')
    try:
        arbitrarydocument = ArbitraryDocumentModel.query.filter_by(title=title).first()
        d = eval(arbitrarydocument.classified_sentences)
        return render_template('seeclassifiedsentences.html',classified_sentences=arbitrarydocument.classified_sentences)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'













@app.route('/companies/')
def companies():
    try:
        tag = request.args.get('tag')
        companies = CompanyDocumentModel.query.all()
        industry_tags = IndustryTags.query.all()
        if(tag and tag!='all'):
            for i in companies[:]:
                if(i.industry_tags is None or tag not in i.industry_tags):
                    companies.remove(i)
    except Exception as e:
        print(e,file=sys.stderr)
    return render_template('companies.html',companies=companies,industry_tags=industry_tags)

@app.route('/companies/addnewcompanies')
def add_new_companies():
    try:
        searchqueries = SearchQueryModel.query.all()
        industrytags = IndustryTags.query.all()
        return render_template('addeditcompanies.html',companydocument=CompanyDocumentModel(title='',clean_text=''),searchqueries=searchqueries,report=None,industrytags=industrytags)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'


@app.route('/companies/editcompany')
def edit_company():
    title = request.args.get('title')
    if(title=='' or title is None):
        return 'no company found'
    try:
        companydocument = CompanyDocumentModel.get_row_by_title(title)
        searchqueries = SearchQueryModel.query.all()
        report = ReportModel.query.filter_by(title='default: '+title).first()
        industrytags = IndustryTags.query.all()
        score = companydocument.query_score
        if(score):
            score = eval(score)
        return render_template('addeditcompanies.html',report=report,companydocument=companydocument,searchqueries=searchqueries,industrytags=industrytags,score=score)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'
@app.route('/companies/deletecompany')
def delete_company():
    title = request.args.get('title')
    if(title=='' or title is None):
        return 'no company found'
    try:
        if(CompanyDocumentModel.delete(title=title)):
            if(ReportModel.query.filter_by(title = 'default: '+title).first()):
                print(11111,file=sys.stderr)
                ReportModel.query.filter_by(title = 'default: '+title).delete()
                db.session.commit()
            return redirect(url_for('companies'))
        else:
            return 'error'
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'

@app.route('/companies/seeclassifiedsentences')
def see_classified_sentences():
    title = request.args.get('title')
    try:
        companydocument = CompanyDocumentModel.get_row_by_title(title)
        d = eval(companydocument.classified_sentences)
        return render_template('seeclassifiedsentences.html',classified_sentences=companydocument.classified_sentences)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'
@app.route('/companies/savecompany',methods=['POST'])
def savecompany():
    try:
        result = request.form
        old_title = result.get('old_title')
        title = result.get('title')
        clean_text = result.get('clean_text')
        industry_tags = result.get('industry_tags')
        reference_to_search_query = result.get('reference_to_search_query')
        run_query_score = result.get('run_query_score')
        if(title is None or title==''):
            return "title cannot be empty"
        if(clean_text is None or clean_text==''):
            return "clean text cannot be empty"
        clean_text = re.sub(r"(\r\n|\r|\n)","",clean_text)
        clean_text = re.sub(r"([^0-9]\.)",r"\1 ",clean_text)

        # if(industry_tags is None or industry_tags==''):
        #     return "industry tags cannot be empty"
        if(reference_to_search_query is None or reference_to_search_query==''):
            return "reference_to_search_query cannot be empty"
        if(reference_to_search_query=='empty'):
            reference_to_search_query=None
        if(run_query_score=='yes'):
            run_query_score = True
        else:
            run_query_score = False
        # temp_clean_text = re.sub(r'[\n]',' ',clean_text)
        # clean_text = re.sub(r'[^a-zA-Z0-9. ]','',temp_clean_text)
        
        # classified_sentences = temp_azure(clean_text)
        res = requests.post(url_for('classified',_external=True), json={"mytext": clean_text})
        if res.ok:
            classified_sentences = str(res.json())
        else:
            classified_sentences = None
        # companydocument = CompanyDocumentModel(title=title,clean_text=clean_text,classified_sentences=str(classified_sentences),industry_tags=industry_tags,reference_to_search_query=reference_to_search_query)
        companydocument = CompanyDocumentModel(title=title,clean_text=clean_text,classified_sentences=str(classified_sentences),reference_to_search_query=reference_to_search_query,industry_tags=industry_tags,run_query_score=run_query_score)
        if(old_title is not None and old_title!=''):
            # CompanyDocumentModel.query.filter_by(title = old_title).update(dict(title=title,clean_text=clean_text,classified_sentences=str(classified_sentences),industry_tags=industry_tags,reference_to_search_query=reference_to_search_query))
            CompanyDocumentModel.query.filter_by(title = old_title).update(dict(title=title,clean_text=clean_text,classified_sentences=str(classified_sentences),reference_to_search_query=reference_to_search_query,industry_tags=industry_tags,run_query_score=run_query_score))
            db.session.commit()
        else:
            db.session.add(companydocument)
            db.session.commit()
        report = ReportModel.query.filter_by(title = 'default: '+title).first()
        if(reference_to_search_query is not None):
            if(run_query_score):
                if(report is None):
                    report = ReportModel(first=title,second=reference_to_search_query,frequency='Weekly',type='vssearchquery',status='running',title='default: '+title,up_to_date=True,range_from=0,range_to=100,dimension='aesthetic',descending=True)
                    db.session.add(report)
                    db.session.commit()
                    executor.submit(report_background,report.id,'vssearchquery',title,reference_to_search_query,0,100,True)
        else:
            try:
                CompanyDocumentModel.query.filter_by(title = old_title).update(dict(query_score=None))
                if(report):
                    ReportModel.query.filter_by(title = 'default: '+title).delete()
                    SentenceModel.delete(f_id=report.id)
                db.session.commit()
            except:
                print('no delete',file=sys.stderr)
        return redirect(url_for('companies'))

    except Exception as e:
        print(e,file=sys.stderr)
        return "Error"





def newdocumentadd(i):
    thread = i.get('thread')
    reach = thread.get('reach')
    social = thread.get('social')
    entities = i.get('entities')
    persons = None
    organizations = None
    locations = None
    if(entities):
        persons = entities.get('persons')
        organizations = entities.get('organizations')
        locations = entities.get('locations')

    newdocument = NewDocumentModel(author=i.get('author'),text=i.get('text'),url=i.get('url'),site=thread.get('site')
                                   ,title=thread.get('title'),title_full=thread.get('title_full')
                                   ,published=thread.get('published'),replies_count=thread.get('replies_count')
                                   ,participants_count=thread.get('participants_count'),site_type=thread.get('site_type')
                                   ,country=thread.get('country'),spam_score=thread.get('spam_score')
                                   ,main_image=thread.get('main_image'),performance_score=thread.get('performance_score')
                                   ,domain_rank=thread.get('domain_rank'))
    if(reach):
        newdocument.reach_per_m = reach.get('per_million')
        newdocument.reach_updated = reach.get('updated')
        page_views = reach.get('page_views')
        if(page_views):
            newdocument.reach_views_per_m = page_views.get('per_million')
            newdocument.reach_views_per_u = page_views.get('per_user')


    if(social):
        facebook = social.get('facebook')
        if(facebook):
            newdocument.facebook_likes = facebook.get('likes')
            newdocument.facebook_comments = facebook.get('comments')
            newdocument.facebook_shares = facebook.get('shares')
        gplus = social.get('gplus')
        if(gplus):
            newdocument.gplus_shares = gplus.get('shares')
        pinterest = social.get('pinterest')
        if(pinterest):
            newdocument.pinterest_shares = pinterest.get('shares')
        linkedin = social.get('linkedin')
        if(linkedin):
            newdocument.linkedin_shares = linkedin.get('shares')
        stumbledupon = social.get('stumbledupon')
        if(stumbledupon):
            newdocument.stumbledupon_shares = stumbledupon.get('shares')
        vk = social.get('vk')
        if(vk):
            newdocument.vk_shares = vk.get('shares')

    db.session.add(newdocument)
    db.session.flush()
    database = []
    if(persons):
        for p in persons:
            temp_p = NewDocumentPersonsModel(f_id=newdocument.id,name=p.get('name'),sentiment=p.get('sentiment'))
            database.append(temp_p)
    if(organizations):
        for o in organizations:
            temp_o = NewDocumentOrganizationsModel(f_id=newdocument.id,name=o.get('name'),sentiment=o.get('sentiment'))
            database.append(temp_o)
    if(locations):
        for l in locations:
            temp_l = NewDocumentLocationsModel(f_id=newdocument.id,name=l.get('name'),sentiment=l.get('sentiment'))
            database.append(temp_l)

    try:
        db.session.add(newdocument)
        for i in database:
            db.session.add(i)
    except:
        db.session.rollback()
    finally:
        db.session.commit()
        db.session.close()

def search_query_documents_background(searchquery):
    try:
        print(searchquery.market_language_code,searchquery.site_type,searchquery.country_code,searchquery.characters,searchquery.site)
        today = datetime.datetime.utcnow().date()
        epoch = (today - datetime.timedelta(days=int(searchquery.freshness))).strftime('%s')
        l = ' language:' + searchquery.market_language_code
        st = ' site_type:' + searchquery.site_type
        c = ' thread.country:' + searchquery.country_code
        ch = ' num_chars:>' + searchquery.characters
        s = ' site:' + searchquery.site
        if(searchquery.market_language_code==''):
           l = ''
        if(searchquery.site_type==''):
           st = ''
        if(searchquery.country_code==''):
           c = ''
        if(searchquery.characters==''):
           ch = ''
        if(searchquery.site==''):
           s = ''

        query_params = {"q": "{}{}{}{}{}{}".format(searchquery.query_string,l,st,c,ch,s),
                        "ts": epoch,
                        "sort": "crawled"}
        print('start')
        print(query_params)
        output = webhoseio.query("filterWebContent", query_params)
        print(output['totalResults'])
    except Exception as e:
        print(1111,e)
    while True:
        for i in output['posts']:
            try:
                try:
                    image = i.get('thread').get('main_image')
                except:
                    image = 'unavailable'
                newdocumentadd(i)
                if i.get('thread'):
                    site = i.get('thread').get('site')
                else:
                    site = ''
                searchquerydocument = SearchQueryDocumentModel(f_title=searchquery.title, title=i.get('title'),
                                                               author=str(i.get('author')),
                                                               provider=str(site),
                                                               url=i.get('url'), image_url=image,
                                                               date=i.get('published'))
                temp_clean_text = i.get('text')
                temp_clean_text = re.sub(r'[\n]', ' ', temp_clean_text)
                temp_clean_text = re.sub(r"([^0-9]\.)", r"\1 ", temp_clean_text)
                searchquerydocument.clean_text = re.sub(r'[^a-zA-Z0-9. ]', '', temp_clean_text)
                res = requests.post('http://13.82.225.206:5000/predict',
                                    json={"mytext": searchquerydocument.clean_text})
                print(3333, file=sys.stderr)
                if res.ok:
                    searchquerydocument.classified_sentences = str(res.json())
                else:
                    searchquerydocument.classified_sentences = None
                if (SearchQueryDocumentModel.query.filter_by(f_title=searchquery.title,url=searchquerydocument.url).first() is None):
                    db.session.add(searchquerydocument)
                    db.session.commit()
                else:
                    print('Already in database', file=sys.stderr)
            except Exception as e:
                print(e, 123, 123, file=sys.stderr)
                db.session.rollback()
                db.session.commit()
                db.session.close()
                continue
        output = webhoseio.get_next()
        if(int(output['moreResultsAvailable'])<1):
            break
    db.session.commit()
    db.session.close()
    print('end')

#### Report section

@app.route('/reports/')
def reports():
    try:
        reports = ReportModel.query.all()
        # d = eval(companydocument.classified_sentences)
        return render_template('reports.html',reports=reports)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'

@app.route('/updatereport',methods=['POST'])
def update_report():
    try:
        result = request.form
        print(result,file=sys.stderr)
        # dimensions = {'aesthetic':0,'craftsmanship':0,'purpose':0,'narrative':0, 'all':0}
        dimensions = {'aesthetic': 0, 'craftsmanship': 0, 'purpose': 0, 'narrative': 0}
        id = result.get('id')
        title = result.get('title')
        descending = result.get('descending')
        range_from = result.get('range_from')
        range_to = result.get('range_to')
        range_from = re.findall('\d+',range_from)
        range_to = re.findall('\d+',range_to)
        if(None in [range_from,range_to]):
            return 'error in range'
        else:
            range_from = int(range_from[0])
            range_to = int(range_to[0])
        if(range_from>=range_to or range_from>100 or range_to>100):
            return 'error in range'
        up_to_date = result.get('up_to_date')
        type = result.get('type')
        dimension = result.get('dimension')
        report = ReportModel.query.filter_by(id=id).first()
        if(report is None):
            return 'no report found'

        if('default: ' in report.title):
            default_company = report.title[9:]
            default_company = CompanyDocumentModel.query.filter_by(title=default_company).first()
        if(descending=='true'):
            descending = True
        else:
            descending = False
        if(up_to_date=='true'):
            up_to_date = True
        else:
            up_to_date = False
        sentences = SentenceModel.query.filter_by(f_id=id).all()
        if(sentences is None or sentences==[]):
            pass
        else:
            for dim in dimensions:
                num = 0
                total = 0
                for sentence in sentences:
                    if(dim!=sentence.dimension):
                        continue
                    total += 1
                    score = sentence.similarity
                    if(score>=range_from and score<=range_to):
                        num+=1
                if(total>0):
                    dimensions[dim] = (num / total) * 100
                else:
                    dimensions[dim] = 0
            dimensions['overall'] = (dimensions['aesthetic']+dimensions['craftsmanship']+dimensions['purpose']+dimensions['narrative']) / 4
            print(dimensions,file=sys.stderr)
        ReportModel.query.filter_by(id=id).update(dict(dimension=dimension,descending=descending,range_from=range_from,range_to=range_to,up_to_date=up_to_date,title=title))
        # ReportModel.query.filter_by(id=id).update(dict(score=str(dimensions),dimension=dimension,descending=descending,range_from=range_from,range_to=range_to,up_to_date=up_to_date,title=title))
        # if('default: ' in report.title):
        #     default_company.query_score = str(dimensions)
        db.session.commit()
        return 'done'
        # d = eval(companydocument.classified_sentences)
        # return render_template('reports.html',reports=reports)
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'
        db.session.rollback()

@app.route('/deletereport')
def delete_report():
    try:
        id = request.args.get('id')
        if(id is None or id==''):
            return 'error'
        if(ReportModel.delete(id=id)):
                SentenceModel.delete(f_id=id)
                return redirect(url_for('reports'))
        else:
            return 'error'
    except Exception as e:
        print(e,file=sys.stderr)
        return 'error'


@app.route('/newreport',methods=['GET','POST'])
def new_report():
    if(request.method=='GET'):
        try:
            type = request.args.get('type')
            companies = CompanyDocumentModel.query.all()
            second = CompanyDocumentModel.query.all()
            second_label = 'Company B'
            if(type==None):
                return render_template('newreport.html',first=companies,type='vscompany',second=companies,second_label=second_label)
            elif(type=='vscompany'):
                pass
            elif(type=='vssearchquery'):
                second = SearchQueryModel.query.all()
                second_label = 'Search Query'
            elif(type=='vstag'):
                second = IndustryTags.query.all()
                second_label = 'Industry Tag'
            else:
                return 'error'
            return render_template('newreport.html',first=companies,type=type,second=second,second_label=second_label)
        except Exception as e:
            print(e,file=sys.stderr)
            return 'error'
    else:
        try:

            result = request.form
            title = result.get('title')
            if(title is None or title=='' or title=='None'):
                return 'title cannot be empty'
            first = result.get('first')
            second = result.get('second')
            frequency = result.get('frequency')
            type = result.get('type')
            up_to_date = result.get('up_to_date')
            if(up_to_date is None or up_to_date is False):
                up_to_date = False
            else:
                up_to_date = True
            report = ReportModel(first=first,second=second,frequency=frequency,type=type,status='running',title=title,up_to_date=up_to_date,range_from=0,range_to=100,dimension='aesthetic',descending=True)
            db.session.add(report)
            db.session.commit()
            executor.submit(report_background,report.id,type,first,second,0,100)
            return redirect(url_for('reports'))

        except Exception as e:
            print(e,file=sys.stderr)
            return 'error'



def get_providers(id,range_from,range_to):
    try:
        providers = []
        temp_sen = SentenceModel.query.filter(SentenceModel.f_id==id,SentenceModel.similarity>=range_from,SentenceModel.similarity<=range_to).all()
        p = {}
        # a = {}
        for i in temp_sen:
            if(p.get(i.provider) is not None):
                p[i.provider] += 1
            else:
                p[i.provider] = 1
            # if(a.get(i.author) is not None):
            #     a[i.author] += 1
            # else:
            #     a[i.author] = 1
        length_sen = len(temp_sen)
        for key,value in p.items():
            providers.append([key,(value/length_sen)*100])
        providers = sorted(providers, key=lambda x: x[1],reverse=True)
        # for key,value in a.items():
        #     authors.append([key,value/length_sen])
        # authors = sorted(authors, key=lambda x: x[1],reverse=True)
        return providers[:5]
    except:
        return 'error'


@app.route('/loadmore',methods=['POST'])
def load_more():
    try:
        id = request.form.get('id')
        if (id is None):
            return 'error'
        offset = request.form.get('offset')
        if (offset is None or offset==''):
            offset = 0
        else:
            offset = int(offset)
        report = ReportModel.query.filter_by(id=id).first()
        if (report is None):
            return 'report not found'
        all_sentences = SentenceModel.query.filter_by(f_id=id, dimension=report.dimension).all()
        sentences = []
        for i in all_sentences:
            if (i.similarity >= report.range_from and i.similarity <= report.range_to):
                sentences.append(i)
        if (len(sentences) > 0):
            sentences = sorted(sentences, key=lambda x: x.similarity, reverse=report.descending)
        d = []
        for i in sentences[offset:offset+20]:
            print(i.sentence1, file=sys.stderr)
            d.append({'sentence1':i.sentence1,'similarity':i.similarity,'sentence2':i.sentence2,'title':i.title})
        return {'data':d}
    except Exception as e:
        return 'error'

@app.route('/reporttest',methods=['GET','POST'])
def report_company_test():
    if(request.method=='GET'):
        try:
            dimensions = ['aesthetic','craftsmanship','purpose','narrative', 'all']
            id = request.args.get('id')
            chartdimension = request.args.get('chartdimension')
            if(chartdimension is None or chartdimension=='' or chartdimension=='None'):
                chartdimension = 'overall'
            if(id is None):
                return 'error'
            report = ReportModel.query.filter_by(id=id).first()
            if(report is None):
                return 'report not found'
            print(1)
            companydocuments = CompanyDocumentModel.query.all()
            print(11)
            searchqueries = SearchQueryModel.query.all()
            print(111)
            tags = IndustryTags.query.all()
            print(1111)
            all_sentences = SentenceModel.query.filter(SentenceModel.f_id==id).all()
            print(11111)
            sentences = []
            for i in all_sentences:
                if(i.dimension==report.dimension and i.similarity>=report.range_from and i.similarity<=report.range_to):
                    sentences.append(i)
            print(11) 
            if(len(sentences)>0):
                sentences = sorted(sentences, key=lambda x: x.similarity,reverse=report.descending)
            type = report.type
            page_url = ''
            score1 = None
            score2 = None
            providers = None
            chartdata = None
            both = []
            print(2)
            # authors = []
            if(type=='vscompany'):
                page_url = 'reportcompanytest.html'
                first_default  =ReportModel.query.filter_by(title='default: '+report.first).first()
                second_default =ReportModel.query.filter_by(title='default: '+report.second).first()
                providers = []
                if(first_default):
                    c1 = CompanyDocumentModel.query.filter_by(title=report.first).first()
                    if(c1 and c1.query_score):
                        score1 = eval(c1.query_score)
                    result1 = get_providers(id=first_default.id,range_from=first_default.range_from,range_to=first_default.range_to)
                    if(result1=='error'):
                        providers.append([])
                    else:
                        providers.append(result1)
                if(second_default):
                    c2 = CompanyDocumentModel.query.filter_by(title=report.second).first()
                    if(c2 and c2.query_score):
                        score2 = eval(c2.query_score)
                    result2 = get_providers(id=second_default.id,range_from=second_default.range_from,range_to=second_default.range_to)
                    if(result2=='error'):
                        providers.append([])
                    else:
                        providers.append(result2)
            elif(type=='vssearchquery'):
                page_url = 'reportsearchquerytest.html'
                providers = []
                if(ReportModel.query.filter_by(title='default: '+report.first).first()):
                    c1 = CompanyDocumentModel.query.filter_by(title=report.first).first()
                    if(c1 and c1.query_score):
                        score1 = eval(c1.query_score)
                #temp_sen = SentenceModel.query.filter(SentenceModel.f_id==report.id,SentenceModel.similarity>=report.range_from,SentenceModel.similarity<=report.range_to).all()
                #temp_sen = SentenceModel.query.filter(SentenceModel.f_id==report.id).all()
                temp_sen = []
                p = {}
                # a = {}
                print(3)

                for i in all_sentences:
                    if i.similarity>=report.range_from and i.similarity<=report.range_to:
                         temp_sen.append(i)


                for i in temp_sen:
                    if(p.get(i.provider) is not None):
                        p[i.provider] += 1
                    else:
                        p[i.provider] = 1
                    # if(a.get(i.author) is not None):
                    #     a[i.author] += 1
                    # else:
                    #     a[i.author] = 1
                length_sen = len(temp_sen)
                for key,value in p.items():
                    providers.append([key,(value/length_sen)*100])
                providers = sorted(providers, key=lambda x: x[1],reverse=True)
                providers = providers[:5]
                print(4)
                # for key,value in a.items():
                #     authors.append([key,value/length_sen])
                # authors = sorted(authors, key=lambda x: x[1],reverse=True)
                print(providers,file=sys.stderr)
                # print(authors,file=sys.stderr)
            elif(type=='vstag'):
                page_url = 'reporttagtest.html'
                # page_url = 'test.html'
                companies_tagged = CompanyDocumentModel.query.all()
                documents_tagged = ArbitraryDocumentModel.query.all()
                both_temp = CompanyDocumentModel.query.filter_by(title=report.first).first()
                if(both_temp and companies_tagged):
                    if(both_temp in companies_tagged):
                        companies_tagged.remove(both_temp)
                    companies_tagged.insert(0,both_temp)
                else:
                    return 'error'
                # if(documents_tagged):
                #     for i in documents_tagged:
                #         if(i.industry_tags and report.second in i.industry_tags):
                #             both.append(i)
                if(companies_tagged):
                    for i in companies_tagged:
                        if(i.industry_tags and report.second in i.industry_tags):
                            if(i.query_score):
                                i.query_score = eval(i.query_score)
                            if(i.query_score):
                                for j in i.query_score:
                                    i.query_score[j] = round(i.query_score[j],2)
                            both.append(i)
            else:
                'error'
            return render_template(page_url,companydocuments=companydocuments,report=report,dimensions=dimensions,sentences=sentences[:20],searchqueries=searchqueries,tags=tags,score1=score1,score2=score2,providers=providers,tagdata=both,chartdimension=chartdimension)

        except Exception as e:
            print(e,file=sys.stderr)
            return 'error'
    else:
        try:
            dimensions = ['aesthetic','craftsmanship','purpose','narrative', 'all']
            result = request.form
            id = result.get('id')
            first = result.get('first')
            second = result.get('second')
            frequency = result.get('frequency')
            type = result.get('type')
            up_to_date = result.get('up_to_date')
            if(up_to_date is None or up_to_date is False):
                up_to_date = False
            else:
                up_to_date = True
            dimension = result.get('dimensions')
            if(dimension is None):
                if(first is None):
                    return 'error'
                report = ReportModel(first=first,second=second,frequency=frequency,type=type,status='incomplete',up_to_date=up_to_date)
                companydocuments = CompanyDocumentModel.query.all()
                searchqueries = SearchQueryModel.query.all()
                if(type=='vscompany'):
                    return render_template('reportcompanytest.html',companydocuments=companydocuments,report=report,new=True,dimensions=dimensions)
                elif(type=='vssearchquery'):
                    return render_template('reportsearchquerytest.html',companydocuments=companydocuments,searchqueries=searchqueries,report=report,new=True,dimensions=dimensions)
                elif(type=='vstag'):
                    pass
                else:
                    return 'error'
            title = result.get('title')
            descending = result.get('descending')
            range_from = result.get('range_from')
            range_to = result.get('range_to')
            range_from = re.findall('\d+',range_from)
            range_to = re.findall('\d+',range_to)
            if(title is None or title==''):
                return 'error in title'
            if(None in [range_from,range_to]):
                return 'error in range'
            else:
                range_from = int(range_from[0])
                range_to = int(range_to[0])
            if(range_from>=range_to or range_from>100 or range_to>100):
                return 'error in range'
            if(descending is None or descending is False):
                descending = False
            else:
                descending = True
            if(id is None or id=='None' or id==''):
                print(123,file=sys.stderr)
                report = ReportModel(first=first,second=second,frequency=frequency,type=type,status='running',dimension=dimension
                ,descending=descending,range_from=range_from,range_to=range_to,title=title,up_to_date=up_to_date)
                db.session.add(report)
                db.session.commit()
                executor.submit(report_background,report.id,type,first,second,range_from,range_to)
                return redirect(url_for('reports'))

            if(ReportModel.query.filter_by(id=id).first() is not None):
                ReportModel.query.filter_by(id = id).update(dict(first=first,second=second,frequency=frequency,type=type,status='running',dimension=dimension
                ,descending=descending,range_from=range_from,range_to=range_to,title=title,up_to_date=up_to_date))
                db.session.commit()
                executor.submit(report_background,id,type,first,second,range_from,range_to)
                return redirect(url_for('reports'))
            else:
                return 'error sqlalchemy'
            db.session.commit()
            return redirect(url_for('reports'))

        except Exception as e:
            print(e,file=sys.stderr)
            return 'error'

def report_background(id,type,first,second,range_from,range_to,default=False):
    try:
        print(id,type,first,second,range_from,range_to,file=sys.stderr)
        dimensions = {'aesthetic':0,'craftsmanship':0,'purpose':0,'narrative':0}
        first = CompanyDocumentModel.query.filter_by(title=first).first()
        dict_company_A = eval(first.classified_sentences)
        if(SentenceModel.query.filter_by(f_id=id).first() is not None):
            if(SentenceModel.delete(f_id=id)):
                pass
            else:
                print('delete error',file=sys.stderr)
        all_sentence1s = []
        all_sentence2s = []
        # note this all_sentences solution to 'all' results in duplicate calculation, but I believe not duplicate database entries
        all_sen_pro_authors = {}
        for dimension in dimensions:
            # dict_query = eval(companydocument_b.classified_sentences)
            sentence1 = []
            sentence2 = []
            sen_pro_author = {}
            for i in dict_company_A:
                if(dict_company_A[i]==dimension):
                    ##sentence1.append(i)
                    if(len(re.findall(r'\w+',i))>3):
                        sentence1.append(i)
                if (len(re.findall(r'\w+', i)) > 3):
                    all_sentence1s.append(i)
            # print(1,sentence1,file=sys.stderr)
            if(type=='vscompany'):
                print("entered vscompany condition")
                second_company = CompanyDocumentModel.query.filter_by(title=second).first()
                print("second company: ", second_company)
                if(second_company.classified_sentences):
                    dict_company = eval(second_company.classified_sentences)
                    #print("dict_company: ", dict_company)
                for i in dict_company:
                    if(dict_company[i]==dimension):
                        ##sentence2.append(i)
                        if(len(re.findall(r'\w+',i))>3):
                            sentence2.append([i,second_company.title,'company'])
                    if (len(re.findall(r'\w+', i)) > 3):
                        all_sentence2s.append([i,second_company.title,'company'])
            elif(type=='vssearchquery'):
                print('entered search query')
                second_searchquery = SearchQueryDocumentModel.query.filter_by(f_title=second).all()
                for querydocument in second_searchquery:
                    if(querydocument.classified_sentences):
                        dict_query = eval(querydocument.classified_sentences)
                    for i in dict_query:
                        if(dict_query[i]==dimension):
                            ##sentence2.append(i)
                            ##sen_pro_author[i] = {'provider':querydocument.provider,'author':querydocument.author}
                            if(len(re.findall(r'\w+',i))>3):
                                sentence2.append([i,querydocument.title,'searchquery'])
                                sen_pro_author[i] = {'provider':querydocument.provider,'author':querydocument.author}
                        if (len(re.findall(r'\w+', i)) > 3):
                            all_sentence2s.append([i,querydocument.title,'searchquery'])
                            all_sen_pro_authors[i] = {'provider': querydocument.provider, 'author': querydocument.author}
            elif(type=='vstag'):
                companies_tagged = CompanyDocumentModel.query.all()
                documents_tagged = ArbitraryDocumentModel.query.all()
                both = []
                if(documents_tagged):
                    for i in documents_tagged:
                        if(i.industry_tags and second in i.industry_tags):
                            both.append(i)
                if(companies_tagged):
                    for i in companies_tagged:
                        if(i.industry_tags and second in i.industry_tags):
                            both.append(i)
                if(first in both):
                    both.remove(first)
                for querydocument in both:
                    if(querydocument.classified_sentences):
                        d = eval(querydocument.classified_sentences)
                    for i in d:
                        if(d[i]==dimension):
                            ##sentence2.append(i)
                            if(len(re.findall(r'\w+',i))>3):
                                sentence2.append([i,querydocument.title,'tag'])
                        if (len(re.findall(r'\w+', i)) > 3):
                            all_sentence2s.append([i,querydocument.title,'tag'])
            else:
                return 'error'
            if(len(sentence1)==0 or len(sentence2)==0):
                continue
            #print(dimension,sentence1,sentence2,file=sys.stderr)
            dimensions[dimension] = get_scores(sentence1,sentence2,dimension,range_from,range_to,id,sen_pro_author)

        # scoring for comparing all sentences
        dimension = 'all'
        get_scores(all_sentence1s, all_sentence2s, dimension, range_from, range_to, id, sen_pro_author=all_sen_pro_authors)

        dimensions['overall'] = (dimensions['aesthetic']+dimensions['craftsmanship']+dimensions['purpose']+dimensions['narrative']) / 4
        ReportModel.query.filter_by(id = id).update(dict(score=str(dimensions),status='done'))
        db.session.commit()
        if('default: ' in ReportModel.query.filter_by(id = id).first().title):
            first.query_score = str(dimensions)
            db.session.commit()

    except Exception as e:
        print(e,file=sys.stderr)
        ReportModel.query.filter_by(id = id).update(dict(status='done'))
        db.session.commit()




def get_scores(sentence1,sentence2,dimension,range_from,range_to,id,sen_pro_author):
    print('get scores')
    #res = requests.post(url_for('sentenceSimliarity',_external=True), json={"sentence1":sentence1,"sentence2":sentence2})
    #print(res.text,file=sys.stderr)
    res_dict = getSimlarity(sentence1,sentence2)
    #if res.ok:
        #res_dict = res.json()
        ##print(res_dict,file=sys.stderr)
    #else:
        #return 'error'
    t=[]
    for i in res_dict:
        for j in res_dict[i]:
            score = abs(float(res_dict[i][j]['similarity'])*100)
            if(sen_pro_author=={}):
                t.append(SentenceModel(sentence1=i,sentence2=j,similarity=int(score),f_id=id,dimension=dimension,title=res_dict[i][j]['title'],type=res_dict[i][j]['type']))
            else:
                t.append(SentenceModel(sentence1=i,sentence2=j,similarity=int(score),f_id=id,dimension=dimension,title=res_dict[i][j]['title'],type=res_dict[i][j]['type'],provider=sen_pro_author.get(j).get('provider'),author=sen_pro_author.get(j).get('author')))
    try:
        db.session.add_all(list(dict.fromkeys(t)))
        db.session.commit()
    except Exception as e:
        print(e,file=sys.stderr)
    num = 0
    total = 0
    threshold = Threshold.query.filter_by(id=1).first()
    for i in res_dict:
        for j in res_dict[i]:
            total += 1
            score = abs(float(res_dict[i][j]['similarity'])*100)
            if(threshold):
                value = threshold.value
            else:
                value = 100
            if(score>=value):
                num+=1
            # if(score>=range_from and score<=range_to):
            #     num+=1
    if(total>0):
        return (num/total)*100
    else:
        return 'error'
def temp_azure(tmp):
    import random
    l = tmp.split('.')
    d = {}
    classes = ['purpose','craftsmanship','aesthetic','narrative']
    for i in l:
        if(len(i)>5):
            d[i+'.'] = random.choice(classes)

    return d
if __name__ == '__main__':
    from waitress import serve
    tl.start()
    serve(app, host='0.0.0.0', port=5000)
    #app.run(
        #host = '0.0.0.0',
        #port = 5000,debug=False
        ##ssl_context = ('cert.pem', 'key.pem')
        #)
