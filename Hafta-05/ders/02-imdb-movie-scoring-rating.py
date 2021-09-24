############################################
# SORTING PRODUCTS - DEVAM
############################################

############################################
# Uygulama: IMDB Movie Scoring & Sorting
############################################


import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv",
                 low_memory=False)  # DtypeWarning kapamak icin
df.head()

df = df[["title", "vote_average", "vote_count"]]


########################
# Vote Average'a Göre Sıralama
########################

df.sort_values("vote_average", ascending=False).head(20)

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T


df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)


########################
# vote_count
########################

df[df["vote_count"] > 400].sort_values("vote_count", ascending=False).head(20)


########################
# vote_average * vote_count
########################

df["average_count_score"] = df["vote_average"] * df["vote_count"]

df.sort_values("average_count_score", ascending=False).head(20)




########################
# weighted_rating
########################

# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)


# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)


# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85



# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66

# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85



M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)


df.sort_values("average_count_score", ascending=False).head(20)

weighted_rating(7.40000, 11444.00000, M, C)

weighted_rating(8.10000, 14075.00000, M, C)

weighted_rating(8.50000, 8358.00000, M, C)

df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(20)


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



# esaretin bedeli (9,2)
bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

# baba (9,1)
bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

# baba2 (9)
bayesian_average_rating([20469, 3892, 4347, 6210, 12657, 26349, 70845, 175492, 324898, 486342])

# karasovalye (9)
bayesian_average_rating([30345, 7172, 8083, 11429, 23236, 49482, 137745, 354608, 649114, 1034843])

# deadpole
bayesian_average_rating([10929, 4248, 5888, 9817, 21897, 59973, 153250, 256674, 197525, 183404])


df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]
df.head()


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)

df.sort_values("bar_score", ascending=False).head(20)