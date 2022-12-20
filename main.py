from collections import Counter
import pandas as pd
import numpy as np
from app import app
from bs4 import BeautifulSoup
import requests
from flask import request, jsonify
import snscrape.modules.twitter as sntwitter
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib


#Separate the previous words with Sastrawi.
factory = StemmerFactory()
stemmer = factory.create_stemmer()
more_stopwords = ['yg', 'kpd', 'utk', 'cuman','hanya','deh', 'btw', 'tapi', 'gua', 'gue', 'lo', 'loe', 'lu',
                  'kalo', 'trs', 'jd', 'nih', 'ntar', 'nya', 'lg', 'yng','ttg','dpt', 'dr', 'kpn', 'on', 
                  'in', 'btw', 'kok', 'kyk', 'donk', 'yah', 'u', 'ya', 'ga', 'gak', 'km', 'eh', 'sih', 
                  'si', 'a', 'b','c','d','e', 'f','g','h','i','j','k','l','m','n','o','p','q','r','s',
                  't','u','v','x','y','z', 'bang', 'bro', 'sob', 'mas', 'mba', 'kmrn', 
                  'iy', 'doang', 'aja', 'iyah', 'lho', 'sbnry', 'tuh', 'kzl', 'ksl', 
                  'weh', 'tuh', 'allahuakbar', 'subhanallah', 'masyaallah', 'rp', 'bbg','gk', 'g', 'ahh',
                 'byebye','an','pd','ah', 'tdk', 'klw', 'tp', 'dll', 'ad','lgi','banget', 'wkwk', 'kwk', 
                  'huhu','jg','oh','emg','omg','klo','ih','gt', 'loh', 'dgn', 'ooh']

stop_words = set(stopwords.words('indonesian') + more_stopwords) 

def clean_text(text):
    text = text.strip().lower()
    text = ' '.join(([t for t in text.split() if not '/' in t]))
    text = ' '.join(([t for t in text.split() if not ('wkwk' in t) 
                      and not ('hehe' in t)
                      and not ('hihi' in t)
                      and not ('haha' in t)
                      and not ('huhu' in t)
                      and not ('xixi' in t)
					  and not ('aa' in t)
                      and not ('ee' in t)
                      and not ('ii' in t)
                      and not ('oo' in t)
                      and not ('uu' in t)]))
    text = re.sub("'","",text)
    text = re.sub('#','hastag',text)
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    text = ' '.join(re.sub(r"(\d)|([A-Za-z0-9]+\d)|(\d[A-Za-z0-9]+)", " ", text).split())
    word_tokens = word_tokenize(text)
    text = ' '.join([t for t in word_tokens if not t in stop_words])
    text_clean = stemmer.stem(text)
    text_clean = re.sub('hastag','#',text_clean)
    return text_clean

count_tweet = joblib.load('count_vect')
transTweet = joblib.load('transformer')
model = joblib.load('model')

count_news = joblib.load('count_vectNews')
transNews = joblib.load('transformerNews')
model_news = joblib.load('modelNews')

months = {'januari':'01','februari':'02','maret':'03','april':'04','mei':'05','juni':'06','juli':'07',
'agustus':'08','september':'09','oktober':'10','november':'11','desember':'12','december':'12'}

# Twitter

@app.route('/twitter', methods=["POST"])
def tweetcrawler():
	keyword = request.json['keyword']
	if request.json['lang'] == '':
		lang = ''
	else:
		lang = ' lang:' + request.json['lang']
	if request.json['until'] == '':
		until = ''
	else:
		until = ' until:' + request.json['until']
	if request.json['since'] == '':
		since = ''
	else:
		since = ' since:' + request.json['since']
	
	query = keyword + lang + until + since
	tweets = []
	limit = request.json['tweet_count']
	for tweet in sntwitter.TwitterSearchScraper(query=query).get_items():
		if len(tweets) == limit:
			break
		else:
			if tweet.inReplyToUser is not None:
				tweet_type = 'reply'
				tweet_reply = re.findall(r'[/]\w+', str(tweet.inReplyToUser))[-1].replace('/','')
			elif tweet.quotedTweet is not None:
				tweet_type = 'retweet'
				tweet_reply = None
			else:
				tweet_type = 'original'
				tweet_reply = None
			if pd.notna(tweet.coordinates):
				latlon = str(tweet.coordinates.latitude) + ',' + str(tweet.coordinates.longitude)
			else:
				latlon = None
			tweets.append({'conversation_id' : tweet.conversationId,
						'coordinate': latlon,
						'date' : tweet.date,
						'hastag' : tweet.hashtags,
						'tweet_id' : tweet.id,
						'reply_totweetid' : tweet.inReplyToTweetId,
						'reply_touserid' : tweet.inReplyToUser,
						'reply' : tweet_reply,
						'lang' : tweet.lang,
						'like_count' : tweet.likeCount,
						'media' : tweet.media,
						'mentioned_user' : tweet.mentionedUsers,
						'outlinks' : tweet.outlinks,
						'quote_count' : tweet.quoteCount,
						'quoted_tweet' : tweet.quotedTweet,
						'content' : tweet.renderedContent,
						'tweet_type' : tweet_type,
						'reply_count ': tweet.replyCount,
						'retweet_count' : tweet.retweetCount,
						'retweeted_tweet' : tweet.retweetedTweet,
						'source_label' : tweet.sourceLabel,
						'url' : tweet.url,
						'user_account' : tweet.user,
						'username' : tweet.user.username})
	resp = jsonify(tweets)
	return resp

@app.route('/sentiment', methods=["POST"])
def sent():
	text = request.json['content']

	new_tweet = np.array([clean_text(text)])
	new_tweet = count_tweet.transform(new_tweet)
	new_tweet = transTweet.transform(new_tweet)
	prediction = model.predict(new_tweet)
	
	sentiment = []

	if prediction == [1]:
		hasil = {'mark':'positif','value':1}
	elif prediction == [0]:
		hasil = {'mark':'netral','value':0}
	else:
		hasil = {'mark':'negatif','value':-1}
	
	sentiment.append(hasil)
	
	resp = jsonify(sentiment)
	return resp

@app.route('/stopword', methods=["POST"])
def stw():
	text = request.json['content']
	text = clean_text(text)
	resp = jsonify(text)
	return resp

@app.route('/wordcloud', methods=["POST"])
def wc():
	content = request.get_json()
	df = pd.DataFrame(content)
	text = []
	for i in df['content']:
		i = clean_text(i)
		j = i.split(' ')
		for k in j:
			text.append(k)
	word_freq = Counter(text)
	resp = jsonify(word_freq)
	return resp

@app.route('/sentimentnews', methods=["POST"])
def sentnews():
	text = request.json['content']

	new_text = np.array([clean_text(text)])
	new_text = count_news.transform(new_text)
	new_text = transNews.transform(new_text)
	prediction = model_news.predict(new_text)
	
	sentiment = []

	if prediction == [1]:
		hasil = {'mark':'positif','value':1}
	elif prediction == [0]:
		hasil = {'mark':'netral','value':0}
	else:
		hasil = {'mark':'negatif','value':-1}
	
	sentiment.append(hasil)
	
	resp = jsonify(sentiment)
	return resp

@app.route('/suaramerdeka', methods=["POST"])
def merdekaNews():
	keyword = request.json['keyword']
	pages = request.json['pages']
	articles = []
	for page in range(1,int(pages)+1):
		merdeka = requests.get('https://www.suaramerdeka.com/search?q={}&page={}'.format(keyword, page))
		beautify = BeautifulSoup(merdeka.content)
		news_list = beautify.find_all('div',{'class','latest__img'})

		for each in news_list:
			try:
				link = each.a.get('href')
				news = requests.get(link)
				soup = BeautifulSoup(news.content)

				date = soup.find('div',{'class', 'read__info__date'}).text.strip()
				date = ' '.join(date.split(' ')[2:-1]).replace('|','').lower()
				for word, replacement in months.items():
					date = date.replace(word, replacement)
				date = pd.to_datetime(date)
				title = soup.find('h1',{'class', 'read__title'}).text.strip()
				text = soup.find('article',{'class', 'read__content clearfix'}).text
				text = re.sub('\n',' ', text).strip()
				text = re.sub(r'\s+',' ', text)
				image = soup.find('div',{'class','photo__img'}).img.get('src')
				portal = 'SUARA MERDEKA'
				
				articles.append({'media': portal,
								'created_at':date,
								'title': title,
								'content': text,
								'url': link,
								'image': image})
				
			except Exception:
				pass

	resp = jsonify(articles)
	return resp

@app.route('/beritajakarta', methods=["POST"])
def beritaNews():
	keyword = request.json['keyword']
	pages = request.json['pages']
	articles = []
	for page in range(1,int(pages)+1):
		beritajakarta = requests.get('https://www.beritajakarta.id/news_index/search?q={}&p={}'.format(keyword, page))
		beautify = BeautifulSoup(beritajakarta.content)
		news_list = beautify.find_all('div',{'class','media blog-list news-index'})

		for each in news_list:
			try:
				link = each.a.get('href')
				news = requests.get(link)
				soup = BeautifulSoup(news.content)

				date = soup.find('div',{'class', 'text-secondary font-italic mb-4'}).text.strip()
				date = ' '.join(date.split(' ')[1:5]).lower()
				for word, replacement in months.items():
					date = date.replace(word, replacement)
				date = pd.to_datetime(date)
				title = soup.find('div',{'class', 'col-7 read mb-4'}).h1.text
				text = soup.find('div',{'class', 'news-content'}).text
				text = re.sub('\n',' ', text).strip()
				text = re.sub(r'\s+',' ', text)
				image = soup.find('div',{'class','col-7 read mb-4'}).img.get('src')
				portal = 'BERITA JAKARTA'
			

				articles.append({'media': portal,
								'created_at':date,
								'title': title,
								'content': text,
								'url': link,
								'image': image})
			except Exception:
				pass

	resp = jsonify(articles)
	return resp

@app.route('/pikiranrakyat', methods=["POST"])
def pikiranNews():
	keyword = request.json['keyword']
	pages = request.json['pages']
	articles = []
	for page in range(1,int(pages)+1):
		pikiran_rakyat = requests.get('https://www.pikiran-rakyat.com/search?q={}&page={}'.format(keyword, page))
		beautify = BeautifulSoup(pikiran_rakyat.content)
		news_list = beautify.find_all('div',{'class','latest__img'})
		
		for each in news_list:
			try:
				link = each.a.get('href')
				news = requests.get(link)
				soup = BeautifulSoup(news.content)

				date = soup.find('div',{'class', 'read__info__date'}).text.strip()
				date = date.split(' ')
				date = ' '.join(date[1:-1]).lower()
				for word, replacement in months.items():
					date = date.replace(word, replacement)
				date = pd.to_datetime(date)
				title = soup.find('h1',{'class', 'read__title'}).text.strip()
				text = soup.find('article',{'class', 'read__content clearfix'}).text.strip()
				text = re.sub('\n',' ', text).strip()
				text = re.sub(r'\s+',' ', text)
				image = soup.find('div',{'class','photo__img'}).img.get('src')
				portal = 'PIKIRAN RAKYAT'

				articles.append({'media': portal,
								'created_at':date,
								'title': title,
								'content': text,
								'url': link,
								'image': image})
			except Exception:
				pass

	resp = jsonify(articles)
	return resp
			

@app.route('/tempo', methods=["POST"])
def tempoNews():
	keyword = request.json['keyword']
	pages = request.json['pages']
	articles = []
	for page in range(1,int(pages)+1):
		tempo = requests.get('https://www.tempo.co/search?q={}&page={}'.format(keyword, page))
		beautify = BeautifulSoup(tempo.content)
		tempo_search = beautify.find_all('div',{'class','card-box ft240 margin-bottom-sm'})

		for each in tempo_search:
			try:
				link = each.a.get('href')
				news = requests.get(link)
				soup = BeautifulSoup(news.content)

				date = soup.find('h4',{'class', 'date margin-bottom-sm'}).text.lower()
				date = date.split(',')[-1].strip()
				date = ' '.join(date.split()[:-1])
				for word, replacement in months.items():
					date = date.replace(word, replacement)
				date = pd.to_datetime(date)
				title = soup.find('h1',{'class', 'title margin-bottom-sm'}).text.strip()
				text = soup.find('div',{'class', 'detail-in'}).text
				text = re.sub('\n',' ', text).strip()
				image = soup.find('div',{'class','foto-detail margin-bottom-sm'}).img.get('src')
				portal = 'TEMPO'

				articles.append({'media': portal,
								'created_at':date,
								'title': title,
								'content': text,
								'url': link,
								'image': image})
			except Exception:
				pass

	resp = jsonify(articles)
	return resp
		
@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status': 404,
        'message': 'Not Found: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 404

    return resp
		
if __name__ == "__main__":
    app.run(debug=True)