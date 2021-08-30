#importing the libraries

import pandas as pd
import re

#importing the dataset

df = pd.read_csv("D:\pkapu\Documents\Sudo-Hack\labeled_data.csv")

#pre-processing of data

from preprocessor.api import clean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
lemmatizer = WordNetLemmatizer()
w_tokenizer = TweetTokenizer()
corpus = []
hashtag=[]
for i in range(0,len(df)):
    hashtag=re.findall(r'#(\w+)', df["tweet"][i])
    show = clean(df["tweet"][i])
    show=re.sub(r'[0-9]+', '', show) #digits
    show=re.sub(r'[^\w\s]', '', show) #puntuation
    show = show.lower()
    show = show.split()
    show = [lemmatizer.lemmatize(word) for word in show if not word in stopwords.words('english')]
    show = ' '.join(show)
    corpus.append(show)
    
#vectorization of data 

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=80,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()
y = df['class'].to_frame()

#train-test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#running the model

from sklearn.naive_bayes import MultinomialNB
nvbs = MultinomialNB()
nvbs.fit(X_train,y_train)
nvbs_score = nvbs.score(X_test,y_test)
print("Accuracy score for NaiveBayes is: ", nvbs_score)

#from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier()
#rfc.fit(X_train,y_train)
#rfc_score = rfc.score(X_test,y_test)
#print("Accuracy score for RandomForestClassifier is: ", rfc_score)

#deployment

def wordopt(show):
    show = clean(df["tweet"][i])
    show=re.sub(r'[0-9]+', '', show) #digits
    show=re.sub(r'[^\w\s]', '', show) #puntuation
    show = show.lower()
    show = show.split()
    show = [lemmatizer.lemmatize(word) for word in show if not word in stopwords.words('english')]
    show = ' '.join(show)
    corpus.append(show)
    return(show)

def output_lable(n):
    if n == 0:
        return "Hate Speech"
    elif n == 1:
        return "Offensive Language"
    elif n == 2:
        return "Neither"

vectorization = TfidfVectorizer()    
def manual_testing(comment):
    testing_comment = {"text":[comment]}
    new_def_test = pd.DataFrame(testing_comment)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_nvbs = nvbs.predict(new_xv_test)

    return print("Prediction: {}".format(output_lable(pred_nvbs[0])))

comment = str(input())
manual_testing(comment)
