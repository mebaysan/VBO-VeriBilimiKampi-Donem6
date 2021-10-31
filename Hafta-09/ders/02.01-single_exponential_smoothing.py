##################################################
# * Single Exponential Smoothing
##################################################
import warnings
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing



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
# * Single Exponential Smoothing
##################################################

# SES = Level
# Durağan serilerde kullanılır.
# Trend ve mevsimsellik varsa kullanılamaz.

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)

y_pred = ses_model.forecast(48) # 48 adımlık (ay) ileriye tahmin yaptık. 48 yapmamızın sebebi test değişkeninin train'den sonraki 48 ayı içeriyor olması.

mean_absolute_error(test, y_pred)


train.plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()

train["1985":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()


def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()


plot_co2(train, test, y_pred, "Single Exponential Smoothing")

ses_model.params # modelin parametreleri

############################
# * Hyperparameter Optimization
############################
# SES'te bir hiperparametremiz vardı. Alfa değeri. Bu değer "geçmiş tahmin edilen değerlere" mi yoksa "geçmiş gerçek değerlere" mi odaklanayım'ı alıyordu.

def ses_optimizer(train, test, alphas, step=48):
    """En iyi alfa değerini bulmayı sağlar.

    Args:
        train (pd.Series): train veri seti
        test (pd.Series): test veri seti
        alphas (list): alfa değerlerinin tutulduğu liste
        step (int, optional): Kaç adım (ay) sonrası tahmin edilecek. Defaults to 48.

    Returns:
        best_alpha, best_mae: En iyi alfa ve mae değerlerini verir
    """
    best_alpha, best_mae = None, float("inf")

    for alpha in alphas: # alfa deeğerlerinin tutulduğu liste içerisinde gez
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha) # döndüğün alfa değeri ile SES model oluştur
        y_pred = ses_model.forecast(step) # parametreden gelen adım kadar sonrasını tahmin et
        mae = mean_absolute_error(test, y_pred) # test seti ile tahmin edilen seti karşılaştır ve mae hesapla

        if mae < best_mae: # en iyi mae'yi işaretle
            best_alpha, best_mae = alpha, mae

        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.8, 1, 0.01)
ses_optimizer(train, test, alphas)

best_alpha, best_mae = ses_optimizer(train, test, alphas)

############################
# * Final SES Model
############################

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha) # yukarıdaki optimizer fonksiyonumdan elde ettiğim best_alpha değeri ile yeni bir model kuruyorum
y_pred = ses_model.forecast(48) # test setimiz train setinden 48 ay sonrayı kapsadığından 48 adım sonrasını forecast ettim

plot_co2(train, test, y_pred, "Single Exponential Smoothing")
mean_absolute_error(test, y_pred)
