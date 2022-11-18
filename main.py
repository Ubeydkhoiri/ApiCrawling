from collections import Counter
import pandas as pd
import numpy as np
from app import app
from flask import request, jsonify
import snscrape.modules.twitter as sntwitter
import string
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from googletrans import Translator
translator = Translator()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

from sklearn.svm import SVC
model = SVC()

	
@app.route('/twitter', methods=["POST"])
def test():
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
			tweets.append({'conversation_id' : tweet.conversationId,
						'coord' : tweet.coordinates,
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
	df = pd.read_csv('https://raw.githubusercontent.com/Ubeydkhoiri/ApiCrawling/main/tweet_sentiments.csv')

	x_train = df['Tweet']
	y_train = df['Label']

	x_train_counts = count_vect.fit_transform(x_train)
	x_train_tfidf = transformer.fit_transform(x_train_counts)

	#model fitting
	model.fit(x_train_tfidf, y_train)

	text = request.json['content']

	translations = translator.translate(text)

	new_tweet = np.array([translations.text])
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

stop_factory = StopWordRemoverFactory().get_stop_words()
more_stopwords = [
	'yg', 'kpd', 'utk', 'cuman', 'deh', 'Btw', 'tapi', 'gua', 'gue', 'lo', 'lu',
	'kalo', 'trs', 'jd', 'nih', 'ntar', 'nya', 'lg', 'gk', 'dpt', 'dr', 'kpn',
	'kok', 'kyk', 'donk', 'yah', 'u', 'ya', 'ga', 'gak', 'km', 'eh', 'sih',
	'bang', 'bro', 'sob', 'mas', 'mba', 'haha', 'wkwk', 'kmrn', 'iy', 'affa',
	'iyah', 'lho', 'sbnry', 'tuh', 'kzl', 'hahaha', 'weh', 'tuh'
]

data = stop_factory + more_stopwords

#Separate the previous words with Sastrawi.
factory = StemmerFactory()
stemmer = factory.create_stemmer()
from nltk.tokenize import TweetTokenizer
happy_emoticons = set([
	':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':x', ':>', '=]', '8)',
	':-D', ':D', ':^)', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D',
	'=D', '=-3', ':-))', ':-)', "-')", ':*', ':^*', '>:P', ':-P', ':P',
	'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)',
	'>:-)', '<3'
])
sad_emoticons = set([
	':L', ':-/', '>:/', ':$', '>:[', ':@', ':-(', ':[', ':-||', '=L',
	':<', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'(", ':\\', ':-c',
	':c', ':(', '>:\\', ':('
])
all_emoticons = happy_emoticons.union(sad_emoticons)

def clean_tweets(tweet):
	tokenizer = TweetTokenizer(preserve_case=False,  
								strip_handles=True,
								reduce_len=True)
	tweet_tokens = tokenizer.tokenize(tweet)
	tweet_clean = []
	for word in tweet_tokens:
		if(word not in data and
			word not in all_emoticons and
			word not in string.punctuation):
				stem_word = stemmer.stem(word)
				tweet_clean.append(stem_word) 
	return ' '.join(tweet_clean)

def clean_text(text):
    text = text.lower()
    text = re.sub('#', 'hastag', text)
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    text = re.sub(r" \d+", "", text)
    stop_words = set(stopwords.words('indonesian'))
    word_tokens = word_tokenize(text)
    text = ' '.join([t for t in word_tokens if not t in stop_words])
    text_clean = re.sub('hastag', '#', text)
    return text_clean

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
		j = i.split(' ')
		for k in j:
			text.append(k)
	data = Counter(text)
	resp = jsonify(data)
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
    app.run(debug=True, port=5001)