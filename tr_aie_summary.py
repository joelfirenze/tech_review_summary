
from bs4 import BeautifulSoup
import pandas as pd

with open(r"ai_e.txt", encoding = 'utf-8') as f:
    soup = BeautifulSoup(f.read())
    
list_1 = soup.find_all("a", attrs={"class":"feed-tz__title__link"})
len(list_1)


clean_list = []
for i in range(len(list_1)):
    link = list_1[i]['href']
    clean_list.append(link)
    i +=1

clean_list

import requests

url = clean_list[13]
res = requests.get(url)
html_page = res.content
soup = BeautifulSoup(html_page, 'html.parser')
text = soup.find_all(text = True)
clean_text = text[text.index("Manage your account"):text.index("Popular")]
article = ' '.join(clean_text)
article


i = 0

while i < 121:
    url = clean_list[i]
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text = True)
    print (str(i), str(text.index('Manage your account'))+",")
    i += 1


#you have to try to see where the numbers might get interrupted



i = 16

while i < 121:
    url = clean_list[i]
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text = True)
    print (str(i), str(text.index('Share'))+",")
    i += 1


i = 88

while i < 121:
    url = clean_list[i]
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text = True)
    print (str(i), str(text.index('Share'))+",")
    i += 1


i = 92

while i < 121:
    url = clean_list[i]
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text = True)
    print (str(i), str(text.index('Share'))+",")
    i += 1

 #you might have to try a few times

 #together with some elbow grease, you generate a csv file


articles = pd.read_csv('articlez2_clean.csv', encoding = "ISO-8859-1")


#the text pre-processing part - importing libraries
import nltk
import numpy as np
import string
import wordcloud
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
import random
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('popular')
nltk.download('vader_lexicon')


nltk.download('punkt')

#more text processing
def split_text_into_paras(text):
  sentences_per_para = 10
  sentences_list = nltk.tokenize.sent_tokenize(text)
  para_list = []
  new_para = ''
  sent_count_per_para = 0
  
  for sentence in sentences_list:
    if (sent_count_per_para < sentences_per_para):
      new_para+=sentence
      sent_count_per_para+=1
    elif(sent_count_per_para==sentences_per_para):
      para_list.append(new_para)
      new_para = ''
      sent_count_per_para = 0
  para_list.append(new_para)
  return para_list


def display_topics(model, feature_names, no_top_word):
  top_20_topics = []
  word_list = []
  for topic_idx, topic in enumerate(model.components_):
    word_list.append(([feature_names[i] for i in topic.argsort()[:-no_top_word - 1:-1]]))
  return (word_list)
    
#declaring stopwords variable, and then adding more stopwords into the existing list
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['newsletter', 'mit', 'subscribe', 'blockchain', 'karen', 'hao', 'will', 'knight', 'technologies'])


len(articles.index)

all_list = []
for i in range(len(articles.index)):
  words = CountVectorizer(max_df=10, min_df=1, max_features=1000, stop_words=stopwords)
  bag_of_words = words.fit_transform(split_text_into_paras(articles.iloc[i]['Article ']))
  word_names = words.get_feature_names()
  lda = LDA(n_components = 2).fit(bag_of_words)
  all_list.append(([i] + display_topics(lda, word_names, 5)))

all_list



import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

stopwords = set(STOPWORDS)
stopwords.update(['newsletter', 'mit', 'subscribe', 'blockchain', 'karen', 'hao', 'will', 'knight', 'technologies', 'say', 'people', 'technology', 'algorithm', 'says', 'one'])

article_text = []
for i in range(len(articles.index)):
    article = articles.iloc[i]['Article ']
    article_text.append(article)

article_text = " ".join(article_text)
#article_text not advisable to do this


#function to calculate cosine similarity 
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

#creating the similarity index
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

#function for splitting apart sentences
def read_article(i):
    article = str(articles.iloc[i]['Article ']).split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences


#actual summariser
def generate_summary(i, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(i)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for x in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[x][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))

# let's begin
generate_summary(0, 2)




