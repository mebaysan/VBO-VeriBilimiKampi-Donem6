############################################
# SORTING REVIEWS
############################################
import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# Up-Down Diff Score = (up ratings) − (down ratings)
###################################################

def score_up_down_diff(up, down):
    return up - down


# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

# Review 1 Score:
score_up_down_diff(600, 400)

# Review 2 Score
score_up_down_diff(5500, 4500)

# Review 1 up yüzdesi nedir? Yüzde 60
# Review 2 up yüzdesi nedir? Yüzde 55

###################################################
# Score = Average rating = (up ratings) / (total ratings)
###################################################


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)


# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

# Review 1 Average Rating Score: 2/2 = 1
# Review 2 Average Rating Score: 100/101 = 0.99


###################################################
# Wilson Lower Bound Score
###################################################

# p = ilgilenen olay / tüm olay
# p için güven aralığı



def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.

    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)

wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)


###################################################
# Case Study:
###################################################

up = [1115, 454, 258, 253, 220, 227, 127, 75, 60, 67, 38, 11, 26, 44, 1, 0, 6, 15, 20]
down = [43, 35, 26, 19, 9, 16, 8, 8, 4, 9, 1, 0, 0, 5, 0, 0, 0, 0, 3]
comments = pd.DataFrame({"up": up, "down": down})

comments["score_up_down_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]), axis=1)

comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"],
                                                                                 x["down"]),
                                                  axis=1)



comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"],
                                                                             x["down"]),
                                                axis=1)

comments.sort_values("wilson_lower_bound", ascending=False)



###################################################
# Case Study
###################################################

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})


# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"], x["down"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)


comments.sort_values("wilson_lower_bound", ascending=False)



# 1 ürün gösterildiği gibi değil. kandırılmayın.
# 2 ürün gösterildiği gibi değil. kandırılmayın.
# 3 ürün gösterildiği gibi değil. kandırılmayın.











