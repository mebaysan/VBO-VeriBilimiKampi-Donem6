##################################################
# * Zaman Serisi Yapısal Analizi
##################################################
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


warnings.filterwarnings('ignore')

############################
# * Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas() # statsmodel altındaki veri setini çekiyoruz. 1958 ile 2001 arasındaki CO2 ölçümlerini içeren bir veri setidir.
y = data.data
y.index.sort_values()

y = y['co2'].resample('MS').mean() # haftalık olan veri setini "aylık" periyota çeviriyorum. Her unique ay için ilgili haftaların ortalamasını alıp atıyoruz

y.head()

y.shape

y.isnull().sum()

y = y.fillna(y.bfill()) # bir sonraki gözlemin değeri ile eksik gözlemi doldur
y.head()

y.plot(figsize=(15, 6))
plt.show()


############################
# * Holdout
############################
# veriyi train ve test olarak 2'ye ayıracağım

# 1958'den 1997'sonuna kadar train set.
train = y[:'1997-12-01']
len(train)  # 478 ay

# 1998'in ilk ayından 2001'in sonuna kadar test set.
test = y['1998-01-01':]
len(test)  # 48 ay

##################################################
# * Zaman Serisi Yapısal Analizi
##################################################


# Durağanlık Testi (Dickey-Fuller Testi)
def is_stationary(y):

    # "HO: Non-stationary"
    # "H1: Stationary"

    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")


is_stationary(y)

# Zaman Serisi Bileşenleri ve Durağanlık Testi
def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if stationary:
        is_stationary(y)

# Toplamsal ve çarpımsal modeller için analiz
for model in ["additive", "multiplicative"]:
    ts_decompose(y, model, stationary=True)


# y(t) = Level + Trend + Seasonality + Noise
# y(t) = Level * Trend * Seasonality * Noise

