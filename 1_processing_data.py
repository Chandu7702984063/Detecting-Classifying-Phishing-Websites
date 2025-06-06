#pip install pandas, scikit-learn, numpy, matplotlib, seaborn, xgboost, lightgbm, tld, googlesearch-python
#conda activate main
#streamlit run 3_retrieving_model_gui.py
#IMPORTING NECESSARY LIBRARIES
from tld import get_tld
import pandas as pd

#DATA LOADING & PREPROCESSING
df=pd.read_csv('malicious_phish.csv')

# print(df.shape)
# print(df.head())
# print(df.type.value_counts())

df = df.replace('benign','safe')
df = df.replace('malware','not safe')
df = df.replace('phishing','not safe')
df = df.replace('defacement','not safe')

# print(df)
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
df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i))#using the function to create a new column in the dataframe

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

df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))

#3. GOOGLE INDEX FUNCTION
#a function to see if the given url is indexed on google
from googlesearch import search
def google_index(url):
    site = search(url, 5)
    return 1 if site else 0

df['google_index'] = df['url'].apply(lambda i: google_index(i))#using the function to create a new column in dataframe

#4. NUMBER OF DOTS FUNCTION
#a function to detect the number of dots(.) in the given url
def count_dot(url):
    count_dot = url.count('.')
    return count_dot

df['count.'] = df['url'].apply(lambda i: count_dot(i))#using the function to create a new column in dataframe

#5. NUMBER OF "WWW" FUNCTION
# a function to detect the number of www's in the given url
def count_www(url):
    url.count('www')
    return url.count('www')

df['count-www'] = df['url'].apply(lambda i: count_www(i))#using the function to create a new column in dataframe

#6. NUMBER OF "@" FUNCTION
#a function to detect the number of @'s in the given url
def count_atrate(url):
    return url.count('@')

df['count@'] = df['url'].apply(lambda i: count_atrate(i))#using the function to create a new column in dataframe

#7. NUMBER OF "/" FUNCTION
#a function to detect the number of /'s in the given url
def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))#using the function to create a new column in dataframe

#8. NUMBER OF "//" FUNCTION
#a function to detect the number of //'s in the given url
def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

df['count_embed_domian'] = df['url'].apply(lambda i: no_of_embed(i))#using the function to create a new column in dataframe

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
df['short_url'] = df['url'].apply(lambda i: shortening_service(i))#using the function to create a new column in dataframe

#10. NUMBER OF 'https' FUNCTION
#a function to detect the number of 'https' in the given url
def count_https(url):
    return url.count('https')
df['count-https'] = df['url'].apply(lambda i : count_https(i))#using the function to create a new column in dataframe

#11. NUMBER OF 'http' FUNCTION
#a function to detect the number of 'http' in the given url
def count_http(url):
    return url.count('http')
df['count-http'] = df['url'].apply(lambda i : count_http(i))#using the function to create a new column in dataframe

#12. NUMBER OF % FUNCTION
#a function to detect the number of %'s in the given url
def count_per(url):
    return url.count('%')
df['count%'] = df['url'].apply(lambda i : count_per(i))#using the function to create a new column in dataframe

#13. NUMBER OF ? FUNCTION
#a function to detect the number of ?'s in the given url
def count_ques(url):
    return url.count('?')
df['count?'] = df['url'].apply(lambda i: count_ques(i))#using the function to create a new column in dataframe

#14. NUMBER OF - FUNCTION
#a function to detect the number of -'s in the given url
def count_hyphen(url):
    return url.count('-')
df['count-'] = df['url'].apply(lambda i: count_hyphen(i))#using the function to create a new column in dataframe

#15. NUMBER OF = FUNCTION
#a function to detect the number of ='s in the given url
def count_equal(url):
    return url.count('=')
df['count='] = df['url'].apply(lambda i: count_equal(i))#using the function to create a new column in dataframe

#16. URL LENGTH FUNCTION
#a function to get the length of the url
def url_length(url):
    return len(str(url))
df['url_length'] = df['url'].apply(lambda i: url_length(i))#using the function to create a new column in dataframe

#17. HOSTNAME LENGTH FUNCTION
#a function to get the hostname length
def hostname_length(url):
    return len(urlparse(url).netloc)
df['hostname_length'] = df['url'].apply(lambda i: hostname_length(i))#using the function to create a new column in dataframe

#18. SUSPICIOUS WORDS FUNCTION
#a function to detect suspicious words, if any.
def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',url)
    if match:
        return 1
    else:
        return 0
df['sus_url'] = df['url'].apply(lambda i: suspicious_words(i))#using the function to create a new column in dataframe

#19. DIGIT COUNTER FUNCTION
#a function to count the number of digits in the given url
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
df['count-digits']= df['url'].apply(lambda i: digit_count(i))#using the function to create a new column in dataframe

#20. LETTER COUNTER FUNCTION
#a function to count the number of letter in the given url
def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters
df['count-letters']= df['url'].apply(lambda i: letter_count(i))#using the function to create a new column in dataframe

#21. TOP LEVEL DOMAIN & LENGTH FUNCTION
#a function to get the first directory length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0
df['fd_length'] = df['url'].apply(lambda i: fd_length(i))#using the function to create a new column in dataframe

#22. FIRST DIRECTORY LENGTH
#a function to get the first directory length
df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))#getting the top level domain using tld library and creating a new column'

def tld_length(tld):#defining a functin to get the length of tld from the column tld creating by above line
    try:
        return len(tld)
    except:
        return -1
df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))#using the function to create a new column in dataframe

#now that we have created all the necessary columns, extracted from url column,
#we will export the data as a csv file and use it later as a pre-processed csv in order to reduce computation time
df = df.drop("tld",axis=1)#dropping the tld column, as it's no longer necessary

print(df.columns)
print(df['type'].value_counts())

df.to_csv('pre-processed-data.csv')