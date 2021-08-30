#importing the libraries
import pandas as pd
import re

#importing the dataset
df = pd.read_csv("D:\pkapu\Documents\Sudo-Hack\labeled_data.csv")

#
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
    
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()

y = df['class'].to_frame()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.naive_bayes import MultinomialNB
nvbs = MultinomialNB()
nvbs.fit(X_train,y_train)
nvbs_score = nvbs.score(X_test,y_test)


