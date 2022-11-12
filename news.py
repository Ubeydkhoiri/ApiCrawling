from app import app
from flask import request, jsonify
from bs4 import BeautifulSoup
import requests
import re


from selenium import webdriver
from bs4 import BeautifulSoup
from newspaper import Article

@app.route('/cnn', methods=["POST"])
def cnnNews():
	keyword = request.json['keyword']
	pages = request.json['pages']
	driver = webdriver.Chrome('C:\selenium\chromedriver.exe')

	articles = []
	for page in range(1, int(pages)+1):
		driver.get("https://www.cnnindonesia.com/search/?query={}&page={}".format(keyword,page))

		page_source = driver.page_source

		soup = BeautifulSoup(page_source, 'html.parser')

		# Berita CNN
		cnn_search = soup.find('div',{'class', 'list media_rows middle'})
		cnn = cnn_search.find_all('article')

		for each in cnn:
			link = each.a.get('href')
			article = Article(link)
			article.download()

			article.authors
			article.parse()

			date = article.publish_date
			title = article.title
			text = article.text
			img = article.top_image

			articles.append({'created_at':date,
							'title':title,
							'content':text,
							'image_url':img})

	resp = jsonify(articles)
	return resp

@app.route('/merdeka', methods=["POST"])
def merdekaNews():
	keyword = request.json['keyword']
	pages = request.json['pages']
	articles = []
	for page in range(1,int(pages)+1):
		merdeka = requests.get('https://www.suaramerdeka.com/search?q={}&page={}'.format(keyword, page))
		beautify = BeautifulSoup(merdeka.content)
		news_list = beautify.find_all('div',{'class','latest__img'})

		for each in news_list:
			link = each.a.get('href')
			news = requests.get(link)
			soup = BeautifulSoup(news.content)
			
			date = soup.find('div',{'class', 'read__info__date'}).text.strip()
			date1 = ' '.join(date.split(' ')[2:]).replace('|',',')
			title = soup.find('h1',{'class', 'read__title'}).text.strip()
			text1 = soup.find('article',{'class', 'read__content clearfix'}).text
			text2 = re.sub('\n',' ', text1).strip()
			text3 = re.sub(r'\s+',' ', text2)
			penulis = soup.find('div',{'class', 'read__info__author'}).text
			image = soup.find('div',{'class','photo__img'}).img.get('src')

			articles.append({'created_at':date1,
							'title': title,
							'content': text3,
							'image_url': image})

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
			link = each.a.get('href')
			news = requests.get(link)
			soup = BeautifulSoup(news.content)

			date = soup.find('div',{'class', 'text-secondary font-italic mb-4'}).text.strip()
			date1 = ' '.join(date.split(' ')[1:5])
			title = soup.find('div',{'class', 'col-8 read mb-4'}).h1.text
			text1 = soup.find('div',{'class', 'news-content'}).text
			text2 = re.sub('\n',' ', text1).strip()
			text3 = re.sub(r'\s+',' ', text2)
			image = soup.find('div',{'class','col-8 read mb-4'}).img.get('src')

			articles.append({'created_at':date1,
							'title': title,
							'content': text3,
							'image_url': image})

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
			link = each.a.get('href')
			news = requests.get(link)
			soup = BeautifulSoup(news.content)

			date = soup.find('div',{'class', 'read__info__date'}).text.strip()
			date1 = date.split(' ')
			date2 = ' '.join(date1[1:])
			title = soup.find('h1',{'class', 'read__title'}).text.strip()
			text1 = soup.find('article',{'class', 'read__content clearfix'}).text.strip()
			text2 = re.sub('\n',' ', text1).strip()
			text3 = re.sub(r'\s+',' ', text2)
			image = soup.find('div',{'class','photo__img'}).img.get('src')
		
			articles.append({'created_at': date2,
						'title': title,
						'content': text3,
						'image_url': image})

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