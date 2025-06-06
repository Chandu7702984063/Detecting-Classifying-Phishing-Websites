#pip install pandas, scikit-learn, numpy, matplotlib, seaborn, xgboost, lightgbm, tld, googlesearch-python
#conda activate main
#streamlit run 3_retrieving_model_gui.py

#IMPORTING NECESSARY LIBRARIES
from tld import get_tld
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

import pickle
from joblib import Parallel, delayed
import joblib
#DATA LOADING & PREPROCESSING
df=pd.read_csv('pre-processed-data.csv')

    #FEATURE ENGINEERING

#1. IP ADDRESS FUNCTION
#creating a function to check if the given url has ip address in it or not. there are 2 types of ip addresses,
#namely, IPv4 and IPv6, hence two lines of code
import re
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0

#2. ABNORMAL URL FUNCTION
from urllib.parse import urlparse
def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        return 1
    else:
        return 0

#3. GOOGLE INDEX FUNCTION
#a function to see if the given url is indexed on google
from googlesearch import search
def google_index(url):
    site = search(url, 5)
    return 1 if site else 0

#4. NUMBER OF DOTS FUNCTION
#a function to detect the number of dots(.) in the given url
def count_dot(url):
    count_dot = url.count('.')
    return count_dot

#5. NUMBER OF "WWW" FUNCTION
# a function to detect the number of www's in the given url
def count_www(url):
    url.count('www')
    return url.count('www')

#6. NUMBER OF "@" FUNCTION
#a function to detect the number of @'s in the given url
def count_atrate(url):
    return url.count('@')

#7. NUMBER OF "/" FUNCTION
#a function to detect the number of /'s in the given url
def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

#8. NUMBER OF "//" FUNCTION
#a function to detect the number of //'s in the given url
def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

#9. SHORTENED URL FUNCTION
#a function to see if the url is shortened
def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

#10. NUMBER OF 'https' FUNCTION
#a function to detect the number of 'https' in the given url
def count_https(url):
    return url.count('https')

#11. NUMBER OF 'http' FUNCTION
#a function to detect the number of 'http' in the given url
def count_http(url):
    return url.count('http')

#12. NUMBER OF % FUNCTION
#a function to detect the number of %'s in the given url
def count_per(url):
    return url.count('%')

#13. NUMBER OF ? FUNCTION
#a function to detect the number of ?'s in the given url
def count_ques(url):
    return url.count('?')

#14. NUMBER OF - FUNCTION
#a function to detect the number of -'s in the given url
def count_hyphen(url):
    return url.count('-')

#15. NUMBER OF = FUNCTION
#a function to detect the number of ='s in the given url
def count_equal(url):
    return url.count('=')

#16. URL LENGTH FUNCTION
#a function to get the length of the url
def url_length(url):
    return len(str(url))

#17. HOSTNAME LENGTH FUNCTION
#a function to get the hostname length
def hostname_length(url):
    return len(urlparse(url).netloc)

#18. SUSPICIOUS WORDS FUNCTION
#a function to detect suspicious words, if any.
def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',url)
    if match:
        return 1
    else:
        return 0

#19. DIGIT COUNTER FUNCTION
#a function to count the number of digits in the given url
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

#20. LETTER COUNTER FUNCTION
#a function to count the number of letter in the given url
def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

#21. TOP LEVEL DOMAIN & LENGTH FUNCTION
#a function to get the first directory length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

#22. FIRST DIRECTORY LENGTH
#a function to get the first directory length
df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))#getting the top level domain using tld library and creating a new column'

def tld_length(tld):#defining a functin to get the length of tld from the column tld creating by above line
    try:
        return len(tld)
    except:
        return -1

#now that we have created all the necessary columns, extracted from url column,
#we will export the data as a csv file and use it later as a pre-processed csv in order to reduce computation time
df = df.drop("tld",axis=1)#dropping the tld column, as it's no longer necessary

#SEGREGATING TARGETS & FEATURES
X = df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
       'count_dir', 'count_embed_domian', 'short_url', 'count-https',
       'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]

#Target Variable
y = df['type']

#splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)

#training the models

    #1. RANDOM FOREST
# Load the model from the file
rf_from_joblib = joblib.load('rf_model.pkl')
# Use the loaded model to make predictions
rf_from_joblib_prediction = rf_from_joblib.predict(X_test)
print(classification_report(y_test,rf_from_joblib_prediction))


    #2. LIGHT GBM CLASSIFIER
# Load the model from the file
lgb_from_joblib = joblib.load('lgb_model.pkl')
# Use the loaded model to make predictions
lgb_from_joblib_prediction = lgb_from_joblib.predict(X_test)
print(classification_report(y_test,lgb_from_joblib_prediction))

#CREATING A STREAMLIT PAGE & TAKING NEW INPUT
import streamlit as st#used for webdev
st.title('URL MALWARE DETECTOR')#setting webpage title
urls = st.text_input('ENTER URL : ')#taking url input
option = st.selectbox('SELECT A MODEL',options=['RANDOM FOREST','LGB CLASSIFIER'])#model selection

submit = st.button('SUBMIT')#submit button
#PREDICTING FOR NEW URLS USING THE TRAINED MODELS
def main(url):#defining a function which takes url as an input, extracts all the necessary features about the url and stores it in an array
    status = []

    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(count_dot(url))
    status.append(count_www(url))
    status.append(count_atrate(url))
    status.append(no_of_dir(url))
    status.append(no_of_embed(url))

    status.append(shortening_service(url))
    status.append(count_https(url))
    status.append(count_http(url))

    status.append(count_per(url))
    status.append(count_ques(url))
    status.append(count_hyphen(url))
    status.append(count_equal(url))

    status.append(url_length(url))
    status.append(hostname_length(url))
    status.append(suspicious_words(url))
    status.append(digit_count(url))
    status.append(letter_count(url))
    status.append(fd_length(url))
    tld = get_tld(url, fail_silently=True)

    status.append(tld_length(tld))
    return status


def get_prediction_from_url(test_url):#creating a function which predicts the output from the feature array creates by main function
    features_test = main(test_url)
    # Due to updates to scikit-learn, we now need a 2D array as a parameter to the predict function.
    features_test = np.array(features_test).reshape((1, -1))

    rf_result = rf_from_joblib.predict(features_test)
    lgb_result = lgb_from_joblib.predict(features_test)
    if option == 'RANDOM FOREST':
        if rf_result[0]=='not safe':
            st.header('RESULT OF RANDOM FOREST MODEL IS : ')
            st.error('THE GIVEN URL IS NOT SAFE TO VISIT')#error == red
        elif rf_result[0] == 'safe':
            st.header('RESULT OF RANDOM FOREST MODEL IS : ')
            st.success('THE GIVEN URL IS SAFE TO VISIT')#success == green
    elif option == 'LGB CLASSIFIER':
        if lgb_result[0] == 'not safe':
            st.header('RESULT OF LGB CLASSIFIER MODEL IS : ')
            st.error('THE GIVEN URL IS NOT SAFE TO VISIT')
        elif lgb_result[0] == 'safe':
            st.header('RESULT OF LGB CLASSIFIER MODEL IS : ')
            st.success('THE GIVEN URL IS SAFE TO VISIT')

if submit:
    urls = [urls]
    for url in urls:
        get_prediction_from_url(url)

    tab1, tab2 = st.tabs(['RANDOM FOREST', 'LGB CLASSIFIER'])
    with tab1:
        st.header('MODEL PARAMETERS')
        st.error("ACCURACY : {0}%".format(round(accuracy_score(y_test, rf_from_joblib_prediction), 2) * 100))
        st.warning("PRECISION : {0}%".format(round(precision_score(y_test, rf_from_joblib_prediction, pos_label='safe'), 2) * 100))
        st.info("F1 SCORE  : {0}%".format(round(f1_score(y_test, rf_from_joblib_prediction, pos_label='safe'), 2) * 100))
        st.success("R2 SCORE  : {0}%".format(round(recall_score(y_test, rf_from_joblib_prediction, pos_label='safe'), 2) * 100))

        st.header('CLASSIFICATION REPORT')
        report = classification_report(y_test, rf_from_joblib_prediction, target_names=['safe', 'not safe'], output_dict=True)#creating a classification report
        report_df = pd.DataFrame(report).transpose()#converting into dataframe
        st.dataframe(report_df, height=212, width=1000)#disaplying dataframe on webpage


    with tab2:
        st.header('MODEL PARAMETERS')
        st.error("ACCURACY : {0}%".format(round(accuracy_score(y_test, lgb_from_joblib_prediction), 2) * 100))
        st.warning("PRECISION : {0}%".format(round(precision_score(y_test, lgb_from_joblib_prediction, pos_label='safe'), 2) * 100))
        st.info("F1 SCORE  : {0}%".format(round(f1_score(y_test, lgb_from_joblib_prediction, pos_label='safe'), 2) * 100))
        st.success("R2 SCORE  : {0}%".format(round(recall_score(y_test, lgb_from_joblib_prediction, pos_label='safe'), 2) * 100))

        st.header('CLASSIFICATION REPORT')
        report = classification_report(y_test, lgb_from_joblib_prediction, target_names=['safe', 'not safe'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, height=212, width=1000)


else:
    if urls=="":
        st.error('EMPTY URL')
    else:
        st.error('ENTER URL & CLICK SUBMIT')

