import numpy as np
from collections import Counter
import csv, pickle, os
from nltk.corpus import stopwords
from flask import current_app as app 
from sentence_transformers import SentenceTransformer, util
import time

# new bert imports
from transformers import TFDistilBertForSequenceClassification, DistilBertConfig,TFDistilBertModel
import tensorflow as tf
from tqdm import tqdm


class_arr = np.array([
    "purpose", 
    "craftsmanship", 
    "aesthetic",
    "narrative"
])
index = {
    "purpose"      : [1, 0, 0, 0], 
    "craftsmanship": [0, 1, 0, 0],  
    "aesthetic"    : [0, 0, 1, 0],
    "narrative"         : [0, 0, 0, 1]
}

#############       DATA PRE-PROCESSING FUNCTIONS       #############   

def load_data(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            temp = row
            break
        
        if row[0].lower() != 'labels' or row[1].lower() != 'sentences':            
            print("ERROR: PLZ NAME THE FIRST ROW 'labels' and 'sentences'")
            return
                
        df = pd.read_csv(path)    
        return df


def count_words(features):
    counter = Counter()
    maximum = 0
    
    for sentence in features:
        maximum = max(maximum, len(sentence))
        
        for word in sentence: 
            counter[word] += 1
            
    return maximum, counter


def filter_func(temp):
    
    stop = set(stopwords.words("english"))
    
    temp = temp.lower()
    temp = temp.split()
    temp = [
        element
        for element in temp
        if element not in stop
    ]
    return temp

filter_func = np.vectorize(filter_func, otypes=[list])    


def shuffle(features, labels):
    
    assert labels.shape[0] == features.shape[0]

    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    
    return features[idx], labels[idx]


def onehot_encode_labels(labels):
    
    return np.array([
        index[e] 
        for e in labels
    ])


def decode_onehot_labels(class_idx):
    
    return class_arr[class_idx] 


#############       LOAD FUNCTIONS      #############

def load_tokenizer():

    with open('static/Pickles/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


def load_classColors():

    with open('static/Pickles/class_colors.pickle', 'rb') as handle:
        class_colors = pickle.load(handle)

    return class_colors


#############       SAVE FUNCTIONS      #############

def save_tokenizer(tokenizer):

    with open('static/Pickles/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_classColors(new_purpose, new_craftsmaship, new_aesthetic, new_none):
    class_colors = {'purpose': new_purpose, 'craftsmaship': new_craftsmaship, 'aesthetic': new_aesthetic, 'narrative':new_none}

    #Overwriting Previous Colors File
    with open('static/Pickles/class_colors.pickle', 'wb') as handle:
        pickle.dump(class_colors, handle, protocol=pickle.HIGHEST_PROTOCOL)


#############       BIN FUNCTIONS      #############


def appendTSVtoBin(labels, sentences):

    with open('bin/output2.tsv', 'a', newline="") as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')

        for i in zip(labels, sentences):
            tsv_writer.writerow([i[0], i[1]])
            print(i[0], i[1])

            
def writeTSVtoBin(labels, sentences):
    with open('bin/output2.tsv', 'w', newline="") as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for i in zip(labels, sentences):
            tsv_writer.writerow([i[0], i[1]])
            print(i[0], i[1]) 
            
            
def loadTSVfromFolder():
    with open("static/File_Upload_Folder/uploaded.tsv", 'r',encoding="utf8") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        try:
            data = [ (row[1], row[0]) for row in tsv_reader ]

        except IndexError as ie:
            print(ie)
            data = []       
    return data               
            
def loadTSVfromBin():

    with open('bin/output2.tsv', 'r',encoding="utf8") as tsv_file:

        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        
        try:
            data = [ (row[1], row[0]) for row in tsv_reader ]

        except IndexError as ie:
            print(ie)
            data = []
            
        
    return data

def clearBin():

    with open('bin/output2.tsv', 'wt') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')

    return


#############       OTHER FUNCTIONS      #############

def singlefile(file):

    list = os.listdir("static/File_Upload_Folder/")
    for i in list:
        os.remove("static/File_Upload_Folder/"+i)

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], "uploaded.tsv"))

def roundoff(arr):

    arr = np.max(arr, axis= 1)
    arr = arr * 100
    arr = np.around(arr, decimals= 3)

    return arr

transformer_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
def getSimlarity(sentence1,sentence2):
  print(sentence1,sentence2)
  #Compute embedding for both lists
  embeddings1 = transformer_model.encode(sentence1, convert_to_tensor=True)
  embeddings2 = transformer_model.encode(sentence2, convert_to_tensor=True)

  #Compute cosine-similarits
  cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
  #Output the pairs with their score
  d = {}
  for i in range(len(sentence1)):
    # s = []
    dd = {}
    for j in range(len(sentence2)):
      dd[sentence2[j]] = "{:.2f}".format(cosine_scores[i][j])
    d[sentence1[i]] = dd
  return d  


#new bert utils ftns
def tokenize(sentences, tokenizer):
    input_ids, input_masks= [],[]
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=128, pad_to_max_length=True, 
                                             return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        #input_segments.append(inputs['token_type_ids'])            
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')    

def embeddings(i,a):
  distil_bert = 'distilbert-base-uncased'

  config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
  config.output_hidden_states = False
  transformer_model = TFDistilBertModel.from_pretrained(distil_bert, config = config)

  #input_ids_in = tf.keras.layers.Input(shape=(128,), name='input_token', dtype='int32')
  #input_masks_in = tf.keras.layers.Input(shape=(128,), name='masked_token', dtype='int32') 

  embedding_layer = transformer_model(i, attention_mask=a)[0]
  cls_token = embedding_layer[:,0,:]
  return cls_token     
