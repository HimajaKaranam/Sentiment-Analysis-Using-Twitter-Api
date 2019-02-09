# myapp.py
from random import random
from bokeh.layouts import layout
from bokeh.layouts import column
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.io import output_file, show
from bokeh.layouts import widgetbox
from bokeh.models import TextInput
import bokeh.layouts
from bokeh.layouts import column, row, Spacer
from bokeh.models.widgets import Dropdown
from twitter import *
import tweepy
import importlib
import pickle
import re
import csv
import numpy
import nltk
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import csv
from collections import defaultdict
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Select
from numpy import pi
from bokeh.plotting import figure
# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#to print tables
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
#import nltk
from sklearn.preprocessing import LabelEncoder
d = defaultdict(LabelEncoder)


#connecting to twitter
auth = tweepy.OAuthHandler("7X3v8oQKlTqStMWLkjy8VRiOQ","Qsmp8qGUJpQVwqSfbSDyf5IqawFS81QCmPuCR2HY2wj6dabza3")
auth.set_access_token("972151015083577346-J6MOkKEvIprvHBD8wvfQWFzWUhtitf2", "DyZav5hZaZoMtVzNcwdwZL8a7tKX35qg2UWNSPhIOzMEF")
api = tweepy.API(auth) 

#loading trained model
f = open('featureList.txt', 'rb')
featureList=pickle.load(f)
f.close()
#f = open('mnbClassifier.pickle', 'rb')
#MNBClassifier=pickle.load(f)
#f.close()
sentiment = ['Positive', 'Negative']

# fitting the vocablary into vectorizer
data = pd.read_csv("Data/genderClassifier.csv", encoding='latin1')
def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

stopWords = getStopWordList('data/stopwords.txt')
data['Tweets'] = [cleaning(s) for s in data['text']]
data['Description'] = [cleaning(s) for s in data['description']]
data = data[data['gender'].notnull()]
data = data[data['gender'] != 'brand']
data = data[data['gender'] != 'unknown']
data = data[data['gender:confidence'] ==1]
data['Tweets'] = data['Tweets'].str.lower().str.split()
data['Tweets'] = data['Tweets'].apply(lambda x : [item for item in x if item not in stopWords])
data['Tweets'] = data['Tweets'].str.join(" ")
data['Description'] = data['Description'].str.lower().str.split()
data['Description'] = data['Description'].apply(lambda x : [item for item in x if item not in stopWords])
data['Description'] = data['Description'].str.join(" ")
vectorizer = CountVectorizer()
encoder = LabelEncoder()
x = vectorizer.fit_transform(data['Description'])
y = encoder.fit_transform(data['gender'])
MNBClassifier=MultinomialNB()
MNBClassifier.fit(x,y)

#start process_tweet
def processTweet(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

def getFeatureVector(tweet,stopWords):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector

def getFeatureString(tweet,stopWords):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return ' '.join(featureVector)

def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end

def analyzeTweets(keyword,api):
    pos_count=0
    count=0
    f = open('my_classifier.pickle', 'rb')
    NBClassifier=pickle.load(f)
    f.close()
    male_pos_count=0
    male_neg_count=0
    female_pos_count=0
    female_neg_count=0
    pos_descp_list=[]
    neg_descp_list=[]
    for tweet in tweepy.Cursor(api.search,
                           q=keyword,
                           result_type="popular",
                           lang="en").items(10):
        count=count+1
        #print(tweet.text)
        processedTestTweet = processTweet(tweet.text)
        processedDescription = processTweet(tweet.user.description)
        processedDescription= getFeatureString(processedDescription,stopWords)
        val=NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet,stopWords)))
        print(" Output value="+val)
        if val=='4':
            pos_count=pos_count+1
            pos_descp_list.append(processedDescription)
        elif val=='0':
            neg_descp_list.append(processedDescription)
    print(pos_descp_list)
    p=vectorizer.transform(pos_descp_list)
    q=vectorizer.transform(neg_descp_list)
    pos_list_output= MNBClassifier.predict(p)
    neg_list_output= MNBClassifier.predict(q)
    if pos_count!=0:
        male_pos_count=(pos_list_output == 1).sum()
    if pos_count==count:
        male_neg_count=(neg_list_output == 1).sum()
    print("Positive male count="+repr(male_pos_count))
    print("Negative male count="+repr(male_neg_count))
    print ("Total positive count="+repr(pos_count))
    print ("Total count="+repr(count))
    print (pos_count/count)
    return [pos_count/count,male_pos_count/count,male_neg_count/count]
    
def analyzeTweetsByPlace(keyword,place,api):
    print('inside analyse place')
    pos_count=0
    count=0
    male_pos_count=0
    male_neg_count=0
    female_pos_count=0
    female_neg_count=0
    pos_descp_list=[]
    neg_descp_list=[]
    places = api.geo_search(query=place, granularity="city")
    place_id = places[0].id
    f = open('my_classifier.pickle', 'rb')
    NBClassifier=pickle.load(f)
    f.close()
    for tweet in tweepy.Cursor(api.search, q=keyword+"&place:%s" % place_id).items(10):
        count=count+1
        processedTestTweet = processTweet(tweet.text)
        val=NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet,stopWords)))
        print(" Output value="+val)
        if val=='4':
            pos_count=pos_count+1
            pos_descp_list.append(tweet.user.description)
        elif val=='0':
            neg_descp_list.append(tweet.user.description)
    p=vectorizer.transform(pos_descp_list)
    q=vectorizer.transform(neg_descp_list)
    pos_list_output= MNBClassifier.predict(p)
    neg_list_output= MNBClassifier.predict(q)
    male_pos_count=(pos_list_output == 1).sum()
    male_neg_count=(neg_list_output == 1).sum()
    print("Positive male count="+repr(male_pos_count))
    print("Negative male count="+repr(male_neg_count))
    print ("Total positive count="+repr(pos_count))
    print ("Total count="+repr(count))
    print (pos_count/count)
    return [pos_count/count,male_pos_count/count,male_neg_count/count]

# create a plot and style its properties
p = figure(x_range=sentiment, y_range=(0,1), plot_height=250, title="sentiment",toolbar_location="above", tools="")
p.height=250
p.width=350

q = figure(x_range=sentiment, y_range=(0,1), plot_height=250, title="sentiment",toolbar_location="above", tools="")
q.height=250
q.width=350

#connect to twitter and login
auth = tweepy.OAuthHandler("7X3v8oQKlTqStMWLkjy8VRiOQ","Qsmp8qGUJpQVwqSfbSDyf5IqawFS81QCmPuCR2HY2wj6dabza3")
auth.set_access_token("972151015083577346-J6MOkKEvIprvHBD8wvfQWFzWUhtitf2", "DyZav5hZaZoMtVzNcwdwZL8a7tKX35qg2UWNSPhIOzMEF")
api = tweepy.API(auth)    

# add a text renderer to our plot (no data yet)
r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="20pt",
           text_baseline="middle", text_align="center")
i = 0
ds = r.data_source

#add other widgets
compareButton = Button(label="Compare",width=120,height=20)
analyzeButton = Button(label="Analyze",width=120)
menu = [("Syracuse", "Syracuse"), ("Chicago", "Chicago"),("California", "California")]
dropdown = Select(title="Location:", value="", options=["Syracuse", "Chicago", "California", ""],width=130)
text1_input = TextInput(value="default", title="Keyword 1:",height=20)
text2_input = TextInput(value="default", title="Keyword 2:",height=20)
l = layout([[text1_input,dropdown,compareButton,text2_input],[p,Spacer(width=250),q],[analyzeButton]],sizing_mode='stretch_both')

# create callbacks for buttons
def analyzeCallback():
    keyword=text1_input.value
    if dropdown.value!="":
        print('Drop down value='+dropdown.value)
        print('Inside if dropdown')
        value=analyzeTweetsByPlace(keyword,dropdown.value,api)
        print('Inside if dropdown')
        plotPieChart(value[0],1-value[0],value[1],value[2],0)
    elif(keyword.strip()!=''):
        value=analyzeTweets(keyword,api)
        plotPieChart(value[0],1-value[0],value[1],value[2],0)
        
def compareCallback():
    keyword1=text1_input.value
    keyword2=text2_input.value
    print('compare call back')
    if dropdown.value!=None:
        value1=analyzeTweetsByPlace(keyword1,dropdown.value,api)
        plotPieChart(value1[0],1-value1[0],value1[1],value1[2],0)
        value2=analyzeTweetsByPlace(keyword2,dropdown.value,api)
        plotPieChart(value2[0],1-value2[0],value2[1],value2[2],2)
        return
    if((keyword1.strip()!='') & (keyword2.strip()!='')):
        value1=analyzeTweets(keyword1,api)
        value2=analyzeTweets(keyword2,api)
        plotPieChart(value1[0],1-value1[0],value1[1],value1[2],0)
        plotPieChart(value2[0],1-value2[0],value2[1],value2[2],2)

# add call back
analyzeButton.on_click(analyzeCallback)
compareButton.on_click(compareCallback)

#add graph functions
def plotPieChart(positive,negative,men_positive,men_negative,num):
    print('inside plot pie')
    sentiment = ['Positive', 'Negative']
    gender= ['male','female']
    color = ["#c9d9d3", "#718dbf"]
    #counts = [a, b]
    data = {'sentiment' : sentiment,
        'male'   : [men_positive,men_negative],
        'female'   : [positive-men_positive,negative-men_negative]}
    source = ColumnDataSource(data=data)
    temp = figure(x_range=sentiment, y_range=(0,1), plot_height=150, title="sentiment",toolbar_location="above", tools="")
    temp.height=250
    temp.width=350
    temp.vbar_stack(gender,x='sentiment', width=0.3, color=color, legend=['Male','Female'], source=source)
    temp.xgrid.grid_line_color = None
    temp.legend.orientation = "horizontal"
    temp.legend.location = "top_center"
    temp.legend.click_policy="hide"
    temp.circle([1,2], [3,4])
    (l.children[1]).children[num]=temp
	
curdoc().add_root(l)