##################################################
# * ARIMA(p, d, q): (Autoregressive Integrated Moving Average)
##################################################
import itertools
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_absolute_error


warnings.filterwarnings('ignore')

############################
# Veri Seti
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
# Holdout
############################

# 1958'den 1997'sonuna kadar train set.
train = y[:'1997-12-01']
len(train)  # 478 ay

# 1998'in ilk ayından 2001'in sonuna kadar test set.
test = y['1998-01-01':]
len(test)  # 48 ay

##################################################
# * ARIMA(p, d, q): (Autoregressive Integrated Moving Average)
##################################################


def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN",
                        title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()


arima_model = ARIMA(train,
                    order=(1, 1, 1) # model derecesi: p, d, q
                    ).fit(disp=0)
arima_model.summary()

y_pred = arima_model.forecast(48)[0]
y_pred = pd.Series(y_pred, index=test.index)


plot_co2(train, test, y_pred, "ARIMA")

arima_model.plot_predict(dynamic=False)
plt.show()


############################
# Hyperparameter Optimization (Model Derecelerini Belirleme)
############################

# 1. AIC İstatistiğine Göre Model Derecesini Belirleme
# 2. ACF & PACF Grafiklerine Göre Model Derecesini Belirleme

############################
# AIC & BIC İstatistiklerine Göre Model Derecesini Belirleme
############################

# p ve q kombinasyonlarının üretilmesi
p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))


def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arma_model_result = ARIMA(train, order).fit(disp=0)
            aic = arma_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params


best_params_aic = arima_optimizer_aic(train, pdq)


############################
# Final Model
############################

arima_model = ARIMA(train, best_params_aic).fit(disp=0)
y_pred = arima_model.forecast(48)[0]
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "ARIMA")


############################
# ACF & PACF Grafiklerine Göre Model Derecesini Belirleme
############################

def acf_pacf(y, lags=30):
    plt.figure(figsize=(12, 7))
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    y.plot(ax=ts_ax)

    # Durağanlık testi (HO: Seri Durağan değildir. H1: Seri Durağandır.)
    p_value = sm.tsa.stattools.adfuller(y)[1]
    ts_ax.set_title(
        'Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    plt.tight_layout()
    plt.show()


acf_pacf(y)


#################################
# ACF Grafiği
#################################


######################
# PACF Grafiği
######################


######################
# ACF ve PACF Grafiği ile Model Derecesi Belirlemek
######################


# ACF genişliği gecikmelere göre "AZALIYORSA" ve PACF p gecikme sonra "KESILIYORSA" AR(p) modeli olduğu anlamına gelir.

# ACF genişliği q gecikme sonra "KESILIYORSA" ve PACF genişliği gecikmelere göre "AZALIYORSA" MA(q) modeli olduğu anlamına gelir.

# ACF ve PACF'nin genişlikleri gecikmelere göre azalıyorsa, ARMA modeli olduğu anlamına gelir.

df_diff = y.diff()
df_diff.dropna(inplace=True)

acf_pacf(df_diff)
