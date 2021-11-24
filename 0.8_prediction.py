import pandas as pd
import random, re, distance, pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import spacy
import numpy as np
from flask import Flask, request, render_template
import xgboost as xgb
pd.set_option('display.max_columns', None)

app = Flask(__name__)

filename = 'model/xgboost_model_96.sav'
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predictQuestions():
    if request.method == "POST":
        
        q1 = request.form["question1"]
        q2 = request.form["question2"]
        print('Input: ',q1, q2)
        prediction = predict(q1,q2)
        return prediction
    #return render_template('index.html')
    return None

def predict(q1,q2):
    
    df = pd.DataFrame()
    df['id'] = [1]

    q1 = q1
    q2 = q2

    df['question1'] = [q1]
    df['question2'] = [q2]

    SAFE_DIV = 0.0001 
    STOP_WORDS = stopwords.words("english")

    def preprocess(x):
        x = str(x).lower()
        x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                            .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                            .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                            .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                            .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                            .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                            .replace("€", " euro ").replace("'ll", " will")
        x = re.sub(r"([0-9]+)000000", r"\1m", x)
        x = re.sub(r"([0-9]+)000", r"\1k", x)
        
        porter = PorterStemmer()
        pattern = re.compile('\W')
        
        if type(x) == type(''):
            x = re.sub(pattern, ' ', x)
        
        if type(x) == type(''):
            x = porter.stem(x)
            example1 = BeautifulSoup(x)
            x = example1.get_text()
        
        return x

    def get_token_features(q1, q2):
        token_features = [0.0]*10
        q1_tokens = q1.split()
        q2_tokens = q2.split()

        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return token_features
        q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
        q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stops.intersection(q2_stops))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
        token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
        token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
        token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
        token_features[7] = int(q1_tokens[0] == q2_tokens[0])
        token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
        token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
        return token_features

    def get_longest_substr_ratio(a, b):
        strs = list(distance.lcsubstrings(a, b))
        if len(strs) == 0:
            return 0
        else:
            return len(strs[0]) / (min(len(a), len(b)) + 1)

    def extract_features(df):
        df["question1"] = df["question1"].fillna("").apply(preprocess)
        df["question2"] = df["question2"].fillna("").apply(preprocess)

        token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
        
        df["cwc_min"]       = list(map(lambda x: x[0], token_features))
        df["cwc_max"]       = list(map(lambda x: x[1], token_features))
        df["csc_min"]       = list(map(lambda x: x[2], token_features))
        df["csc_max"]       = list(map(lambda x: x[3], token_features))
        df["ctc_min"]       = list(map(lambda x: x[4], token_features))
        df["ctc_max"]       = list(map(lambda x: x[5], token_features))
        df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
        df["first_word_eq"] = list(map(lambda x: x[7], token_features))
        df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
        df["mean_len"]      = list(map(lambda x: x[9], token_features))
        df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
        df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
        df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
        df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
        df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
        return df

    df = extract_features(df)

    df['freq_qid1'] = [random.randint(1,50)]
    df['freq_qid2'] = [random.randint(1,50)]

    df['q1len'] = df['question1'].str.len()
    df['q2len'] = df['question2'].str.len()

    df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
    df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))

    def normalized_word_Common(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1 * len(w1 & w2)

    df['word_Common'] = df.apply(normalized_word_Common, axis=1)

    def normalized_word_Total(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1 * (len(w1) + len(w2))

    df['word_Total'] = df.apply(normalized_word_Total, axis=1)

    def normalized_word_share(row):
        w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
        
    df['word_share'] = df.apply(normalized_word_share, axis=1)

    df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
    df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])

    df2 = df.copy()

    df2 = df2[['question1','question2']]

    questions = list(df2['question1']) + list(df2['question2'])

    tfidf = TfidfVectorizer(lowercase=False)
    tfidf.fit_transform(questions)
    word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
    nlp = spacy.load('en_core_web_sm')

    vecs1 = []
    for qu1 in tqdm(list(df2['question1'])):
        doc1 = nlp(qu1) 
        mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
        for word1 in doc1:
            vec1 = word1.vector
            try:
                idf = word2tfidf[str(word1)]
            except:
                idf = 0
            mean_vec1 += vec1 * idf
        mean_vec1 = mean_vec1.mean(axis=0)
        vecs1.append(mean_vec1)

    df2['q1_feats_m'] = list(vecs1)

    vecs2 = []
    for qu2 in tqdm(list(df2['question2'])):
        doc2 = nlp(qu2) 
        mean_vec2 = np.zeros([len(doc2), len(doc2[0].vector)])
        for word2 in doc2:
            vec2 = word2.vector
            try:
                idf = word2tfidf[str(word2)]
                print("idf: ", idf)
            except:
                idf = 0
            mean_vec2 += vec2 * idf
        mean_vec2 = mean_vec2.mean(axis=0)
        vecs2.append(mean_vec2)

    df2['q2_feats_m'] = list(vecs2)

    df = df.drop(['question1','question2'],axis=1)

    df3 = df2.drop(['question1','question2'],axis=1)
    df3_q1 = pd.DataFrame(df3.q1_feats_m.values.tolist(), index= df3.index)
    df3_q2 = pd.DataFrame(df3.q2_feats_m.values.tolist(), index= df3.index)

    df3_q1['id']=df['id']
    df3_q2['id']=df['id']
    dfm1  = df.merge(df3_q1, on='id',how='left')
    dfm2  = dfm1.merge(df3_q2, on='id',how='left')

    dfm2.drop(['id'], axis=1, inplace=True)
    dfm2.to_csv("PredictionInput.csv")

    dfm2 = pd.read_csv('PredictionInput.csv')
    dfm2.drop(['Unnamed: 0'], axis=1, inplace=True)

    d_test = xgb.DMatrix(dfm2)

    prediction = loaded_model.predict(d_test)
    print(prediction)

    if prediction > 0.90:
        print("duplicate questions")
        result = 'Duplicate questions'
    else:
        print("non duplicate questions")
        result = 'Non-Duplicate questions'

    return result

if __name__ == '__main__':
    app.run(port=5001,debug=True)