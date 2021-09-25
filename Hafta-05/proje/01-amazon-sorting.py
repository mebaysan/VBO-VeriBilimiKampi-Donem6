import pandas as pd
import numpy as np
import math
import scipy.stats as st


raw_data = pd.read_csv('Datasets/amazon_review.csv')

df = raw_data.copy()

df.head()

df['overall'].mean() # Ortalama rating

def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)



def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df['helpful_no'] = df['total_vote'] - df['helpful_yes']

df['up_down_diff_score'] = df.apply(lambda x: score_up_down_diff(x['helpful_yes'],x['helpful_no']), axis=1)
df['up_down_average_score'] = df.apply(lambda x: score_average_rating(x['helpful_yes'],x['helpful_no']), axis=1)


df['up_down_wilson_lower_bond_score'] = df.apply(lambda x: score_average_rating(x['helpful_yes'],x['helpful_no']), axis=1)


df.sort_values('up_down_diff_score',ascending=False).head(20) # Faydalı - Faydasız oranına göre ilk 20
df.sort_values('up_down_average_score',ascending=False).head(20) # Faydalı yorum sayısı oranına göre ilk 20

df.sort_values('up_down_wilson_lower_bond_score',ascending=False).head(20) # * İstatistiksel olarak ilk 20