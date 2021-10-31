##################################################
# * Triple Exponentıal Smoothing (Holt-Winters)
##################################################
import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


warnings.filterwarnings('ignore')

############################
# * Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data
y.index.sort_values()

y = y['co2'].resample('MS').mean()

y.head()

y.shape

y.isnull().sum()

y = y.fillna(y.bfill())
y.head()

y.plot(figsize=(15, 6))
plt.show()


############################
# * Holdout
############################

# 1958'den 1997'sonuna kadar train set.
train = y[:'1997-12-01']
len(train)  # 478 ay

# 1998'in ilk ayından 2001'in sonuna kadar test set.
test = y['1998-01-01':]
len(test)  # 48 ay

##################################################
# * Triple Exponentıal Smoothing (Holt-Winters)
##################################################
# TES = SES + DES + Mevsimsellik
# TES modeli durağanlık testi vs bakmadan kullanabiliriz. Bu metod leveli, trendi, mevsimselliği yakalayabilir.

def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()


tes_model = ExponentialSmoothing(train,
                                 trend="add", # add || mul
                                 seasonal="add", # add || mul
                                 seasonal_periods=12 # 1 yılda (12, 12 step, 12 adım) 1 mevsimsel periyot tanımlanıyor dedik
                                 ).fit(smoothing_level=0.5, # alpha
                                        smoothing_slope=0.5, # beta
                                        smoothing_seasonal=0.5 # gamma
                                        )


y_pred = tes_model.forecast(48)

plot_co2(train, test, y_pred, "Triple Exponential Smoothing")


############################
# * Hyperparameter Optimization
############################

def tes_optimizer(train, test, abg, trend_mode='add', seasonal_mode = 'add', seasonal_period=12,step=48):
    """TES modeli için hiperparametreleri optimize eden fonksiyon

    Args:
        train (pd.Series): train veri seti
        test (pd.Series): test veri seti
        abg (list): alpha, beta, gamma değerlerini aynı tuple içerisinde tutan tuple'lardan oluşan liste
        trend_mode (str, optional): 'add' || 'mul' . Trend eklemeli mi yoksa çarpmalı mı. Defaults to 'add'.
        seasonal_mode (str, optional): 'add' || 'mul' . Mevsimsellik eklemeli mi yoksa çarpmalı mı. Defaults to 'add'.
        seasonal_period (int, optional): Bir mevsimsel periyodu belirleyen adım sayısı. Defaults to 12.
        step (int, optional): Train seti ile oluşturulan model kaç adım sonrasını forecast edecek. Defaults to 48.

    Returns:
        [float, float, float, float]: best_alpha, best_beta, best_gamma, best_mae
    """
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")

    for comb in abg: # her kombinasyonu gez
        tes_model = ExponentialSmoothing(train, trend=trend_mode, seasonal=seasonal_mode, seasonal_periods=seasonal_period).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2]) # 0: alpha, 1: beta, 2: gamma. Her kombinasyondan gelen değerler ile bir TES model oluştur
        y_pred = tes_model.forecast(step) # oluşturduğun model ile step adım kadar sonrasını forecast et
        mae = mean_absolute_error(test, y_pred) # mae hesapla
        if mae < best_mae: # en iyi parametreleri işaretle
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae


alphas = betas = gammas = np.arange(0.10, 1, 0.20)
abg = list(itertools.product(alphas, betas, gammas)) # 3 listeninde kombinasyonlarını oluştur


best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train,test, abg)


############################
# * Final TES Model
############################

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma) # yukarıdaki fonksiyondan gelen en iyi hiperparametreler ile model kur

y_pred = final_tes_model.forecast(48)

plot_co2(train, test, y_pred, "Triple Exponential Smoothing")


