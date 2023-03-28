from os import listdir
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from statistics import stdev,mean

import warnings
warnings.filterwarnings("ignore")

def load_article(file_name):
    file = open(file_name, encoding = 'utf-8')
    text = file.read()
    file.close()
    return text

def split_article_story_highlight(article):
    index = article.find('@highlight')
    story, highlights = article[: index], article[index :].split('@highlight')
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights

def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


data_path = "/CNN/stories"
csv_path = "/CNN/CNN_with_summary.csv"
clean_csv_path = "/CNN/data_cleaned1.csv"
stories = list()
for name in tqdm(listdir(data_path)):
    filename = data_path + '/' + name
    article = load_article(filename)
    story, highlights = split_article_story_highlight(article)
    stories.append({'story': story, 'highlights': highlights})
print('Loaded Stories %d' % len(stories))
cnn_df = pd.DataFrame.from_dict(stories)
cnn_df.columns = ['article', 'summary']
cnn_df.to_csv(csv_path, index = False)
CNN = pd.read_csv(csv_path)
CNN = pd.read_csv(csv_path)
article_text = []
for i in CNN.article.values:
    new_text = re.sub(r'\n', ' ', i)
    new_text = re.sub(r'(CNN)', ' ', new_text)
    new_text = re.sub(r'LRB', ' ', new_text)
    new_text = re.sub(r'RRB', ' ', new_text)
    new_text = re.sub(r'<', ' ', new_text)
    new_text = re.sub(r'>', ' ', new_text)
    new_text = re.sub(r'[" "]+', " ", new_text)
    new_text = re.sub(r'-- ', ' ', new_text)
    new_text = re.sub(r"([?!¿])", r" \1 ", new_text)
    new_text = re.sub(r'-', ' ', new_text)
    new_text = re.sub(r'\s+', ' ', new_text)
    new_text = re.sub('[^A-Za-z0-9.,]+', ' ', new_text)
    new_text = decontracted(new_text)
    new_text = new_text.replace('/', ' ')
    new_text = new_text.lower()
    article_text.append(new_text)
data_article = pd.DataFrame(article_text, columns = ['Article'])
summary_text = []
for i in CNN.summary.values:
    new_text = re.sub(r'\n',' ', i)
    new_text = re.sub(r'(CNN)', ' ', new_text)
    new_text = re.sub(r'LRB', ' ', new_text)
    new_text = re.sub(r'RRB', ' ', new_text)
    new_text = re.sub(r'>', ' ', new_text)
    new_text = re.sub(r'<', ' ', new_text)
    new_text = re.sub(r'-', ' ', new_text)
    new_text = re.sub(r'[" "]+', " ", new_text)
    new_text = re.sub(r'-- ', ' ', new_text)
    new_text = re.sub(r'\s+', ' ', new_text)
    new_text = re.sub('[^A-Za-z0-9.]+', ' ', new_text)
    new_text = new_text.replace('/', ' ')
    new_text = decontracted(new_text)
    new_text = new_text.lower()
    summary_text.append(new_text)
summary_text = np.array(summary_text)
summary_text = summary_text.reshape(-1, 1)
data_summ = pd.DataFrame(summary_text, columns = ['Summary'])
data_cleaned = data_article.join(data_summ)
data_cleaned.to_csv(clean_csv_path, index = False)
