############################################
# SORTING PRODUCTS
############################################

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama


# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Diğer Faktorler
# - Uygulama: Kurs Sıralama
# - Uygulama: IMDB Movie Scoring & Sorting

# Sorting Reviews
# - Score Up-Down Diff
# - Average rating
# - Wilson Lower Bound Score
# - Uygulama: E-Ticaret Ürün Yorumlarının Sıralanması


###################################################
# Rating Products
###################################################

############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6


df = pd.read_csv("datasets/course_reviews.csv")
df.head()

df["Rating"].value_counts()

df["Questions Asked"].value_counts()


df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})

df.head()

####################
# Average
####################

df["Rating"].mean()



####################
# Time-Based Weighted Average
####################

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

current_date = pd.to_datetime('2021-02-10 0:0:0')

df["days"] = (current_date - df['Timestamp']).dt.days

df.head()


df.loc[df["days"] <= 30, "Rating"].mean() * 28 / 100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26 / 100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24 / 100 + \
df.loc[(df["days"] > 180), "Rating"].mean() * 22 / 100

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)


####################
# User-Based Weighted Average
####################

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df)


####################
# Weighted Rating
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w / 100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)


###################################################
# Sorting Products
###################################################

###################################################
# Uygulama: Kurs Sıralama
###################################################

df = pd.read_csv("datasets/product_sorting.csv")
df.head()

####################
# Sorting by Rating
####################

df.sort_values("rating", ascending=False).head(20)

####################
# Sorting by Comment Count or Purchase Count
####################

df.sort_values("purchase_count", ascending=False).head(20)

df.sort_values("commment_count", ascending=False).head(20)

####################
# Sorting by Rating, Comment and Purchase
####################

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df["commment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

df.head()

(df["commment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["commment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

# df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)


####################
# Bayesian Average Rating Score
####################

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


def bayesian_average_rating(n, confidence=0.95):
    """
    Olasılıksal

    Parameters
    ----------
    n: list or df
        puanların frekanslarını tutar.
        Örnek: [2, 40, 56, 12, 90] 2 tane 1 puan, 40 tane 2 puan, ... , 90 tane 5 puan.
    confidence: float
        güven aralığı

    Returns
    -------
    BAR score: float


    """

    # rating'lerin toplamı sıfır ise sıfır dön.
    if sum(n) == 0:
        return 0
    # eşsiz yıldız sayısı. 5 yıldızdan da puan varsa 5 olacaktır.
    K = len(n)
    # 0.95'e göre z skoru.
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    # toplam rating sayısı.
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    # index bilgisi ile birlikte yıldız sayılarını gez.
    # formülasyondaki hesapları gerçekleştir.
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df["bar_sorting_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                        "2_point",
                                                                        "3_point",
                                                                        "4_point",
                                                                        "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

df.sort_values("bar_sorting_score", ascending=False).head(20)


####################
# Hybrid Sorting: BAR Score + Diğer Faktorler
####################


def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

# df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)

























