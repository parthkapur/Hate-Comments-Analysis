#importing the libraries

import pandas as pd
import re

#importing the dataset

df = pd.read_csv("D:\pkapu\Documents\Sudo-Hack\labeled_data.csv")

#pre-processing of data

from preprocessor.api import clean
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")
def clean_tweets(df):
  tempArr = []
  for line in df:
    # send to tweet_processor
    tmpL = clean(line)
    # remove puctuation
    tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
    tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
    tempArr.append(tmpL)
  return tempArr

train_tweet = clean_tweets(df["tweet"])
#train_tweet = pd.DataFrame(train_tweet).tolist()
df["clean_tweet"] = train_tweet

#vectorization of data 

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3),lowercase=False)

X=tfidf_v.fit_transform(train_tweet)
y = df['class'].to_frame()

#train-test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

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

def output_lable(n):
    if n == 0:
        return "Hate Speech"
    elif n == 1:
        return "Offensive Language"
    elif n == 2:
        return "Neither"

vectorization = TfidfVectorizer()    
def manual_testing(comment):
    testing_comment = {"tweet":[comment]}
    new_def_test = pd.DataFrame(testing_comment)
    new_def_test["tweet"] = new_def_test["tweet"].apply(clean_tweets) 
    new_x_test = new_def_test["tweet"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_nvbs = nvbs.predict(new_xv_test)
    return print("Prediction: {}".format(output_lable(pred_nvbs[0])))

comment = str(input())
manual_testing(comment)