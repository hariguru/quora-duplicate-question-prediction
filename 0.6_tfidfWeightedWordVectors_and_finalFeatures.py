import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys, os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# exctract word2vec vectors
# https://github.com/explosion/spaCy/issues/1721
# http://landinghub.visualstudio.com/visual-cpp-build-tools
import spacy

# avoid decoding problems
df = pd.read_csv("dataset/train.csv")
 
# encode questions to unicode
# https://stackoverflow.com/a/6812069
# ----------------- python 2 ---------------------
# df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
# df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))
# ----------------- python 3 ---------------------
df['question1'] = df['question1'].apply(lambda x: str(x))
df['question2'] = df['question2'].apply(lambda x: str(x))

df.head()
""" 
   id  qid1  qid2                                          question1  \
0   0     1     2  What is the step by step guide to invest in sh...
1   1     3     4  What is the story of Kohinoor (Koh-i-Noor) Dia...
2   2     5     6  How can I increase the speed of my internet co...
3   3     7     8  Why am I mentally very lonely? How can I solve...
4   4     9    10  Which one dissolve in water quikly sugar, salt...

                                           question2  is_duplicate
0  What is the step by step guide to invest in sh...             0
1  What would happen if the Indian government sto...             0
2  How can Internet speed be increased by hacking...             0
3  Find the remainder when [math]23^{24}[/math] i...             0
4            Which fish would survive in salt water?             0 """

# merge texts
questions = list(df['question1']) + list(df['question2'])

tfidf = TfidfVectorizer(lowercase=False)
tfidf.fit_transform(questions)

# dict key:word and value:tf-idf score
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
""" 
sample:
'Qnet': 12.20514086571145, 'QoQ': 13.909888957949875, 'Qorvo': 13.909888957949875, 'Qoura': 11.169048934024675, 'Qr': 13.909888957949875, 'Qraft': 13.21674177738993, 'Qrcode': 13.909888957949875, 'Qs': 13.21674177738993, 'Qt': 12.811276669281765, 'Qu': 13.909888957949875, 

- After we find TF-IDF scores, we convert each question to a weighted average of word2vec vectors by these scores.
- here we use a pre-trained GLOVE model which comes free with "Spacy". https://spacy.io/usage/vectors-similarity
- It is trained on Wikipedia and therefore, it is stronger in terms of word semantics."""

# en_vectors_web_lg, which includes over 1 million unique vectors.
nlp = spacy.load('en_core_web_sm')

vecs1 = []
# https://github.com/noamraph/tqdm
# tqdm is used to print the progress bar
for qu1 in tqdm(list(df['question1'])):
    doc1 = nlp(qu1) 
    # 96 is the number of dimensions of vectors 
    mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
    for word1 in doc1:
        # word2vec
        vec1 = word1.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word1)]
        except:
            idf = 0
        # compute final vec
        mean_vec1 += vec1 * idf
    mean_vec1 = mean_vec1.mean(axis=0)
    vecs1.append(mean_vec1)
df['q1_feats_m'] = list(vecs1)

vecs2 = []
for qu2 in tqdm(list(df['question2'])):
    doc2 = nlp(qu2) 
    mean_vec2 = np.zeros([len(doc1), len(doc2[0].vector)])
    for word2 in doc2:
        # word2vec
        vec2 = word2.vector
        # fetch df score
        try:
            idf = word2tfidf[str(word2)]
        except:
            idf = 0
        # compute final vec
        mean_vec2 += vec2 * idf
    mean_vec2 = mean_vec2.mean(axis=0)
    vecs2.append(mean_vec2)
df['q2_feats_m'] = list(vecs2)

#prepro_features_train.csv (Simple Preprocessing Feartures)
#nlp_features_train.csv (NLP Features)
if os.path.isfile('dataset/nlp_features_train.csv'):
    dfnlp = pd.read_csv("dataset/nlp_features_train.csv",encoding='latin-1')
else:
    print("generate nlp_features_train.csv from 0.5_advanced_EDA.py file")

if os.path.isfile('dataset/df_fe_without_preprocessing_train.csv'):
    dfppro = pd.read_csv("dataset/df_fe_without_preprocessing_train.csv",encoding='latin-1')
else:
    print("generate df_fe_without_preprocessing_train.csv from 0.4_exploratory_data_analysis.py file")

df1 = dfnlp.drop(['qid1','qid2','question1','question2'],axis=1)
df2 = dfppro.drop(['qid1','qid2','question1','question2','is_duplicate'],axis=1)
df3 = df.drop(['qid1','qid2','question1','question2','is_duplicate'],axis=1)
df3_q1 = pd.DataFrame(df3.q1_feats_m.values.tolist(), index= df3.index)
df3_q2 = pd.DataFrame(df3.q2_feats_m.values.tolist(), index= df3.index)

# dataframe of nlp features
df1.head()
""" 
   id  is_duplicate   cwc_min   cwc_max   csc_min   csc_max   ctc_min  \
0   0             0  0.999980  0.833319  0.999983  0.999983  0.916659
1   1             0  0.799984  0.399996  0.749981  0.599988  0.699993
2   2             0  0.399992  0.333328  0.399992  0.249997  0.399996
3   3             0  0.000000  0.000000  0.000000  0.000000  0.000000
4   4             0  0.399992  0.199998  0.999950  0.666644  0.571420

    ctc_max  last_word_eq  first_word_eq  abs_len_diff  mean_len  \
0  0.785709           0.0            1.0           2.0      13.0
1  0.466664           0.0            1.0           5.0      12.5
2  0.285712           0.0            1.0           4.0      12.0
3  0.000000           0.0            0.0           2.0      12.0
4  0.307690           0.0            1.0           6.0      10.0

   token_set_ratio  token_sort_ratio  fuzz_ratio  fuzz_partial_ratio  \
0              100                93          93                 100
1               86                63          66                  75
2               63                63          43                  47
3               28                24           9                  14
4               67                47          35                  56

   longest_substr_ratio
0              0.982759
1              0.596154
2              0.166667
3              0.039216
4              0.175000 """

# data before preprocessing 
df2.head()
"""    id  freq_qid1  freq_qid2  q1len  q2len  q1_n_words  q2_n_words  \
0   0          1          1     66     57          14          12
1   1          4          1     51     88           8          13
2   2          1          1     73     59          14          10
3   3          1          1     50     65          11           9
4   4          3          1     76     39          13           7

   word_Common  word_Total  word_share  freq_q1+q2  freq_q1-q2
0         10.0        23.0    0.434783           2           0
1          4.0        20.0    0.200000           5           3
2          4.0        24.0    0.166667           2           0
3          0.0        19.0    0.000000           2           0
4          2.0        20.0    0.100000           4           2 """

# Questions 1 tfidf weighted word2vec
df3_q1.head()
""" 
           0           1           2           3          4          5   \
0   78.682992   87.635912   77.898819  -61.473692  44.053226  18.525178
1   99.993008   55.174564   -2.049167   36.677249  85.412371 -45.989080
2   62.709638   72.489519   10.889310  -45.772860  71.261772 -34.385969
3   35.006791  -40.413219   53.450493  -45.069038  37.137247 -21.992808
4  135.425154  187.445625  143.612776 -111.735024  56.977977 -70.101866

          6           7           8           9           10         11  \
0 -28.609312   47.452460  -86.095610   58.907952   -5.908887   5.062267
1  31.112590   76.453094  -74.456509  110.348369   84.611222 -41.494358
2 -26.228285   18.224490 -113.496336  115.968702   38.446388   6.606776
3 -28.184323  131.916699   41.891510   27.243861  -81.736980  48.172197
4 -47.585533   59.575895  -56.992457  253.326808 -123.363763  74.103075

           12         13         14          15          16          17  \
0  136.384631 -29.311246  -6.908221 -104.905819  106.754770   16.799190
1   34.701861  10.153459  -8.266818  -23.137087   83.302971   49.561717
2   74.324362  24.936682  34.229029   41.480330   -6.715438  -35.863293
3  -83.910213 -36.551715  30.696786   92.143373   49.486582 -138.234718
4  192.244374 -23.418808  -0.072270  -52.374237   88.718404    6.200053

          18          19          20         21         22          23  \
0 -23.684501   76.534100  113.292545  35.874052  48.696723  -11.653754
1 -54.129983   28.575020  109.895182  85.913793  14.017500 -100.408989
2   8.603262  106.322444   -4.632774   0.949820  92.368805  -62.332417
3 -22.038837   -4.167189  -19.487544 -30.428707  46.449947   81.127514
4 -34.748398  118.860650  104.257678 -19.308273  -5.911030 -130.655260

           24          25          26          27         28          29  \
0  -36.541520   99.915375 -142.438337  -72.028596  -1.374175 -129.174994
1  -73.572298  116.830950 -166.630144 -143.004758  -0.770938  -97.216777
2  -58.682641  145.766957  -50.016893 -108.476287 -25.467648  -48.748818
3  104.318424  121.269531  -28.571857    5.302182 -61.549565  -18.612830
4  -23.421555  125.433206 -133.497143 -179.988439  67.516360  -76.359375

           30         31          32          33         34          35  \
0  -25.437618 -42.146922   37.407022  -69.269756 -73.936123  -19.662360
1  -53.970125 -68.309437  181.591619  -92.624022 -17.250550  147.139629
2    6.924941   6.448761   23.785554  -21.784350 -62.634957    5.436183
3   -8.255657 -59.045547   16.189935   39.718045 -52.527881  -58.007130
4 -150.262275   4.916667   -9.865799 -147.885067  -7.246085  -40.686417

          36         37         38         39          40         41  \
0  38.209890  -9.545738 -44.947935 -56.545574  103.351133 -41.773517
1  87.249062  -0.182777 -34.705845  21.572029   92.693047 -15.391700
2 -13.409507   9.269375  -3.466993 -60.888072  102.830437   8.143691
3  29.170820 -23.750451  73.603838  24.064038   34.067341 -13.916953
4 -92.136775 -73.312214  29.648440  25.839449  266.192679  -4.070586

          42          43          44         45         46          47  \
0  -1.615270  -86.685536  179.614234 -32.611244  51.479851    8.153700
1 -17.352219  -51.056824   69.353060 -32.944922  54.505842   79.090806
2 -34.012816  -27.113287  161.985490 -95.804729 -74.151134   82.334418
3 -28.489230   -1.064873   88.392229  53.934564 -49.078122   30.161458
4  -7.189907 -101.205866  205.504083 -56.870189  80.091094  108.538138

           48         49          50         51          52         53  \
0  -72.668160 -14.568319  -42.159194 -64.791713 -114.053887 -35.884712
1  -80.186894  22.345294  -50.556837 -30.395256 -102.353237 -96.444680
2  -18.426852  -8.371303  -80.706157 -29.161945  -71.158556  -7.246943
3   56.377043  -5.712807  -65.776705 -46.446562   22.449847  56.491043
4 -102.876120  16.201895 -116.265227 -35.394072  -80.900812 -34.551696

           54          55          56         57          58          59  \
0 -126.372452  233.877351  119.046858 -43.580892 -129.956927  -17.886887
1 -142.156060   83.219254   21.462435 -41.674029  -45.288238   25.713024
2  -78.805472  216.249159   17.782252 -41.108650  -91.765161   39.959913
3   55.362357   31.832603   66.836505  67.236734 -136.054820   -2.969362
4  -83.786475  247.585437   12.143946 -58.540757 -112.478575  127.197083

           60          61         62          63          64          65  \
0   74.544202  -84.923952  12.030664 -105.147784  -52.161516  128.408010
1  -46.048228  -13.792427  64.060821 -148.117162   14.021728  222.104868
2   65.969145  -67.563025  18.809625  -42.309049 -118.789139  137.138837
3  114.634710 -109.762412  15.616244   90.188304   -2.994025   13.949361
4  -68.109813  -45.138518   7.509471 -123.735964  -72.285366  189.937254

           66         67         68         69          70          71  \
0   62.392989 -85.797754  75.066873  68.268576  -26.459104   60.869161
1   60.743130  13.882971  92.245109  52.623690  -94.634880   71.679861
2   81.460901 -47.298543  39.536516  56.302137   36.771198   39.998114
3   26.629185 -33.313318  47.752399 -69.337008  -23.252295  -12.457546
4  157.002570 -74.696703  82.379056  76.553088 -125.760949  110.094822

           72         73         74         75          76         77  \
0  122.432411  25.912285  76.802800  52.803958  122.555440  20.075123
1   20.999988 -59.635796  64.200658 -17.341305   36.679837 -32.517277
2   87.886053 -27.177739  -2.240621  56.019091   86.501850   4.331870
3   10.267123   3.404362  73.453422  13.668431   84.808816 -21.767701
4  125.219594  -4.329686 -47.015737  54.651598  147.590678  24.471602

          78          79          80         81         82          83  \
0 -38.572492  -28.747160  -62.788667 -43.180552   2.760628   -7.026932
1 -77.499462    3.254362  -11.735660 -16.854048 -88.464100  -39.774370
2 -55.073498   11.235763  -48.251050  -9.591573 -14.177249  -28.909442
3 -37.630875   33.004449 -112.147202 -13.459578 -18.140102  -58.453283
4 -61.596534  103.159982 -100.474603 -72.754540  30.080280 -115.924926

           84         85         86         87         88         89  \
0 -140.634527 -29.274653   7.157678  86.842601  38.238606 -25.909486
1  -15.653514  66.209611 -64.475704  27.344039 -22.471263 -23.111044
2  -48.646742 -19.231418  77.008272   5.414788 -26.222928  35.709896
3  -10.981707 -14.289388 -36.975472  25.987250 -74.511655 -45.798322
4  -85.053259 -79.693878 -34.521699  74.533560  -3.963831 -77.077944

          90         91          92         93          94         95
0   3.169638 -54.031532 -112.663659   1.619508   81.309565 -17.361949
1 -97.185489  13.815928  -24.577477  72.654378   58.654857 -19.836278
2 -49.750098 -74.032807 -130.011004 -84.557644   10.153947 -30.314630
3  42.739461 -17.318146   37.957786 -47.867102 -101.418604  -3.919247
4  27.673524 -87.661703 -146.777092   1.730535    5.950078 -12.494797 """

# Questions 2 tfidf weighted word2vec
df3_q2.head()

print("Number of features in nlp dataframe :", df1.shape[1])
print("Number of features in preprocessed dataframe :", df2.shape[1])
print("Number of features in question1 w2v  dataframe :", df3_q1.shape[1])
print("Number of features in question2 w2v  dataframe :", df3_q2.shape[1])
print("Number of features in final dataframe  :", df1.shape[1]+df2.shape[1]+df3_q1.shape[1]+df3_q2.shape[1])

# storing the final features to csv file
if not os.path.isfile('dataset/final_features_96.csv'):
    df3_q1['id']=df1['id']
    df3_q2['id']=df1['id']
    df1  = df1.merge(df2, on='id',how='left')
    df2  = df3_q1.merge(df3_q2, on='id',how='left')
    result  = df1.merge(df2, on='id',how='left')
    result.to_csv('dataset/final_features_96.csv')


