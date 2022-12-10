from app import app
from flask import request, jsonify
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import numpy as np
import joblib

from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
factory = StemmerFactory()
stemmer = factory.create_stemmer()

more_stopwords = ['a','b','c','d','a', 'b','c','d','e', 'f','g','h','i','j','k','l','m','n','o','p','q','r','s',
                  't','u','v','x','y','z','ala']

stop_words = set(stopwords.words('indonesian') + more_stopwords)

def clean_text(text):
    text = text.strip().lower()
    text = re.sub("'","",text)
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    text = ' '.join(re.sub(r"(\d)|([A-Za-z0-9]+\d)|(\d[A-Za-z0-9]+)", " ", text).split())
    word_tokens = word_tokenize(text)
    text = ' '.join([t for t in word_tokens if not t in stop_words])
    text_clean = stemmer.stem(text)
    return text_clean

count_vect = joblib.load('count_vectNews')
transformer = joblib.load('transformerNews')
model = joblib.load('modelNews')

months = {'januari':'01','februari':'02','maret':'03','april':'04','mei':'05','juni':'06','juli':'07',
'agustus':'08','september':'09','oktober':'10','november':'11','desember':'12','december':'12'}

@app.route('/sentimentnews', methods=["POST"])
def sent():
	text = request.json['content']

	new_tweet = np.array([clean_text(text)])
	new_tweet = count_vect.transform(new_tweet)
	new_tweet = transformer.transform(new_tweet)
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

@app.route('/stopwordnews', methods=["POST"])
def stw():
	text = request.json['content']
	text = clean_text(text)
	resp = jsonify(text)
	return resp

@app.route('/wordcloudnews', methods=["POST"])
def wc():
	content = request.get_json()
	df = pd.DataFrame(content)
	text = []
	for i in df['content']:
		j = i.split(' ')
		for k in j:
			text.append(k)
	data = Counter(text)
	resp = jsonify(data)
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

		try:
			for each in news_list:
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

		try:
			for each in news_list:
				link = each.a.get('href')
				news = requests.get(link)
				soup = BeautifulSoup(news.content)

				date = soup.find('div',{'class', 'text-secondary font-italic mb-4'}).text.strip()
				date = ' '.join(date.split(' ')[1:5]).lower()
				for word, replacement in months.items():
					date = date.replace(word, replacement)
				date = pd.to_datetime(date)
				title = soup.find('div',{'class', 'col-8 read mb-4'}).h1.text
				text = soup.find('div',{'class', 'news-content'}).text
				text = re.sub('\n',' ', text).strip()
				text = re.sub(r'\s+',' ', text)
				image = soup.find('div',{'class','col-8 read mb-4'}).img.get('src')
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
		
		try:
			for each in news_list:
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

		try:
			for each in tempo_search:
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