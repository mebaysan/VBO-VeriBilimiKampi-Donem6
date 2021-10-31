##################################################
# * Double Exponential Smoothing (DES)
##################################################
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
# * Double Exponential Smoothing (DES)
##################################################

# DES: Level (SES) + Trend
# Level'a ek olarak Trendi yakalayabiliyor.

def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()



# y(t) = Level + Trend + Seasonality + Noise -> DES için Toplamsal trend (trend = 'add')
# y(t) = Level * Trend * Seasonality * Noise -> DES için Çarpımsal trend (trend = 'mul')
# Mevsimsellik ve artık bileşenleri trendden bağımsız ise toplamsal bir seri
# Mevsimsellik ve artık bileşenleri trende bağımlıysa yani trende göre şekilleniyorsa çarpımsal bir seri.
# ######################
# Peki serinin çarpımsal mı toplamsal mı olduğuna nasıl karar veririm?
# - Serinin bileşenlerini inceleyerek bu durumu gözlemleyebilirim
# - En düşük hatayı hangisi veriyorsa o dur. İşe makine öğrenmesi tarafından baktığımız için model hatası rmse veya mae hangisi düşük veriyorsa onunla kurmayı düşünebilirim.


des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5,
                                                         smoothing_trend=0.5)

y_pred = des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")

des_model.params


############################
# * Hyperparameter Optimization
############################


def des_optimizer(train, test, alphas, betas, step=48, trend_mode='add'):
    """DES modeli için aldığı alpha ve beta listelerinin hepsini deneyerek minimum hatayı veren en iyi hiperparametreleri verir

    Args:
        train (pd.Series): train seti
        test (pd.Series): test seti
        alphas (list): denenecek alfa değerleri (level)
        betas (list): denenecek beta değerleri (trend)
        step (int, optional): Kaç adım sonrası için forecast yapılacak (test setindeki ile eşit olmalı). Defaults to 48.
        trend_mode (str, optional): Trend'i eklemeli veya çarpımsal olarak hesaplar. Parametreler = ['add', 'mul']. Defaults to 'add'.

    Returns:
        [float, float, float]: best_alpha, best_beta, best_mae
    """
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas: # her bir alfa değeri
        for beta in betas: # her alfa gezdiğinde bütün betaları gez
            des_model = ExponentialSmoothing(train, trend=trend_mode).fit(smoothing_level=alpha,
                                                                     smoothing_slope=beta) # eğitim veri seti ile bir DES model kuruyorum. Döngüdeki alfa ve beta değerleri ile.
            y_pred = des_model.forecast(step) # step adım kadar sonrası için forecast yapıyorum.
            mae = mean_absolute_error(test, y_pred) # mae hesaplanıyor.
            if mae < best_mae: # en iyi mae'yi veren alpha ve beta değerleri işaretleniyor
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae


alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

best_alpha, best_beta, best_mae = des_optimizer(train, test, alphas, betas)


############################
# * Final DES Model
############################

final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha,
                                                               smoothing_slope=best_beta) # yukarıdaki fonksiyondan elde ettiğim en iyi hiperparametreler ile yeni bir final model kuruyorum

y_pred = final_des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")

