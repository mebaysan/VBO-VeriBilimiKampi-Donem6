##################################################
# Introduction to Text Mining and Natural Language Processing
##################################################

##################################################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
##################################################


# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling

# !pip install nltk
# !pip install textblob
# !pip install wordcloud

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
############################
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import Word, TextBlob
from wordcloud import WordCloud


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


##################################################
# 1. Text Preprocessing
##################################################

df = pd.read_csv("datasets/amazon_review.csv", sep=",")
df.head()
df.info()

###############################
# Normalizing Case Folding
###############################
# bütün harfleri küçülttüm
df['reviewText'] = df['reviewText'].str.lower()

###############################
# Punctuations
###############################
# noktalama işaretlerinden kurtuluyorum
df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')

###############################
# Numbers
###############################
# sayılardan kurtuluyorum
df['reviewText'] = df['reviewText'].str.replace('\d', '')

###############################
# Stopwords
###############################
# ölçüm değeri taşınmayacak olan kelimelerin silinmesi

# nltk.download('stopwords')
# >>> from nltk import download
# >>> download('stopwords')
sw = stopwords.words('english')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

###############################
# Rarewords
###############################
# biraz daha custom bir ihtiyaçtır.

drops = pd.Series(' '.join(df['reviewText']).split()).value_counts() # her kelimenin kaç adet geçtiği
drops = drops[drops <= 1] # 1 kere geçen kelimeleri bir listeye alıyorum

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in drops)) # ilgili review içerisindeki kelimelerden`drops` içinde olmayanlar

###############################
# Tokenization
###############################
# metinleri parçalarına (kelimelere) ayırmaktır

# nltk.download("punkt")
# >>> from nltk import download
# >>> download("punkt")
df["reviewText"].apply(lambda x: TextBlob(x).words).head()

###############################
# Lemmatization
###############################
# Kelimeleri köklerine ayırma işlemidir. ÖR: "göz" den türemiş kelimeler: gözlük, gözcü vb. Ölçüm niteliğini artırmak için kelimeleri köklerine ayırma işlemi gerçekleştiriyoruz
# nltk.download('wordnet')
# >>> from nltk import download
# >>> download("wordnet")
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df['reviewText'].head(10)

##################################################
# 2. Text Visualization
##################################################

###############################
# Terim Frekanslarının Hesaplanması
###############################
# hangi kelimeden kaç adet var
tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]
tf.head()
tf.shape
tf["words"].nunique()
tf["tf"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

###############################
# Barplot
###############################
# kelime frekanslarından bar plot çizelim
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

###############################
# Wordcloud
###############################
# kelime bulutu oluşturuyorum
text = " ".join(i for i in df.reviewText)
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# daha açık renkli bir grafik
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")

###############################
# Şablonlara Göre Wordcloud
###############################
# kelime bulutunun arkasına resim koyabiliriz
vbo_mask = np.array(Image.open("notes/HAFTA_11/img/tr.png"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=vbo_mask,
               contour_width=3,
               contour_color="firebrick")

wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

wc.to_file("vbo.png")

##################################################
# 3. Sentiment Analysis
##################################################
# NLTK already has a built-in, pretrained sentiment analyzer
# called VADER (Valence Aware Dictionary and sEntiment Reasoner).

df.head()
# nltk.download('vader_lexicon')
# >>> from nltk import download
# >>> download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("The film was awesome") # ilgili string ifadenin duygu skoru
sia.polarity_scores("I liked this music but it is not good as the other one") # -1 ile +1 arasında değer alır.
# +1'e yaklaştıkça pozitif, -1'e yaklaştıkça negatif

# mesela review'ları büyültmek istersek:
df["reviewText"].apply(lambda x: x.upper())

# şimdi skorları hesaplayalım mesela 10 tanesi için
df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))

# peki bu sözlük içerisinden sadece bir bileşeni seçmek istersek ne yapacağız?
df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

# işlemi kalıcı olarak yapalım:
df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])
df.head()


##################################################
# 4. Sentiment Modeling
##################################################

###############################
# Feature Engineering
###############################

###############################
# Target'ın Oluşturulması
###############################

df["reviewText"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg") # polarity score 0'dan büyük ise pos değil ise neg olarak etiketleyeceğiz

# şimdi tüm veri için aynı işlemi yapıp veri setinin içine sentiment_label adında bir değişken ekleyelim:
df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df.head(20)

# dengesiz veri problemimiz var mı bir sınıf dağılımına bakalım
df["sentiment_label"].value_counts()

# bir soru daha merak ettiğim şey şu verilen puanlar açısından neg-pos labelleri arasında fark var mı?
df.groupby("sentiment_label")["overall"].mean()

# target'ın encode edilmesi
df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

X = df["reviewText"]
y = df["sentiment_label"]

# Count Vectors: frekans temsiller
# TF-IDF Vectors: normalize edilmiş frekans temsiller
# Word Embeddings (Word2Vec, GloVe, BERT vs)
#      Ağırlık öğrenmeye benzer şekilde kelime vector'lerinin öğrenilmesiyle oluşturulan sayısal temsiller


###############################
# NGrams
###############################
a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim.
N-gram'lar birlikte kullanılan kelimelerin kombinasyolarını gösterir ve feature üretmek için kullanılır"""
# ngram
TextBlob(a).ngrams(3)

# Metinden N kadar kelimeyi kaydırarak sıralar
# ÖR:
# >>> Bu örneği anlaşılabilmesi
# >>> örneği anlaşılabilmesi için
# >>> anlaşılabilmesi için daha


###############################
# Count Vectors
###############################

from sklearn.feature_extraction.text import CountVectorizer
# matris oluşturur => [hangi metinde] X [hangi kelimeden kaç tane var]

corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']

# word frekans
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
X_c.toarray()

# n-gram frekans
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X_n = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names()
X_n.toarray()


# Veriye uygulanması:
vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X)

vectorizer.get_feature_names()[10:15]
X_count.toarray()[10:15]


###############################
# TF-IDF
###############################
# word tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word')
X_w = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
X_w.toarray()

# n-gram tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_n = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
X_n.toarray()


# Veriye uygulanması:
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)

# Veriye uygulanması:
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_word_vectorizer.fit_transform(X)


###############################
# 5. Modeling
###############################

###############################
# Logistic Regression
###############################

log_model = LogisticRegression().fit(X_tf_idf_word, y)

cross_val_score(log_model,
                X_tf_idf_word,
                y, scoring="accuracy",
                cv=5).mean()

yeni_yorum = pd.Series("this product is great")
yeni_yorum = pd.Series("look at that shit very bad")
yeni_yorum = pd.Series("it was good but I am sure that it fits me")

yeni_yorum = CountVectorizer().fit(X).transform(yeni_yorum)
log_model.predict(yeni_yorum)

# orjinal yorumlardan modele sorabilir miyiz?
random_review = pd.Series(df["reviewText"].sample(1).values)
random_review

yeni_yorum = CountVectorizer().fit(X).transform(random_review)
log_model.predict(yeni_yorum)


###############################
# Random Forests
###############################

# Count Vectors
rf_model = RandomForestClassifier().fit(X_count, y)
cross_val_score(rf_model, X_count, y, cv=5, n_jobs=-1).mean()

# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean()

# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
cross_val_score(rf_model, X_tf_idf_ngram, y, cv=5, n_jobs=-1).mean()

###############################
# Hiperparametre Optimizasyonu
###############################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            verbose=True).fit(X_count, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_count, y)

cv_results = cross_validate(rf_final, X_count, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
