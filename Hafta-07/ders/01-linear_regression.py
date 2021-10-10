######################################################
# Sales Prediction with Linear Regression
######################################################

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)


##########################
# * Linear Regression Sürecini Anlamak
##########################

# Rastgele b ve w oluşturmak
bs = np.random.randint(100, 300, 10)
ws = np.random.randint(1, 10, 10)

b = np.random.choice(bs)
w = np.random.choice(ws)

# bağımlı değişken
y = pd.Series([300, 310, 310, 330, 340, 350, 350, 400, 420, 450, 450, 470])

# bağımsız değişken
x = pd.Series([70, 73, 75, 80, 80, 80, 82, 83, 85, 90, 92, 94])

# b ve w kullanarak tahmin edilen değerler
y_pred = b + x * w

# 70 metrekare evin fiyatı
print(b, w)
b + w * 70

# gercek ve tahmin edilen değerlerin bir araya getirilmesi
df = pd.DataFrame({"x": x, "y": y, "y_pred": y_pred})


# hata metriklerine göre hatalar.

def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


# mse
mse(df["y"], df["y_pred"])

# rmse
rmse(df["y"], df["y_pred"])

# mae
mae(df["y"], df["y_pred"])

######################################################
# * Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("datasets/Advertising.csv")
df.shape
df.head()

X = df[["TV"]]
y = df[["sales"]]

##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# sabit (b - bias)
reg_model.intercept_[0]

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]

##########################
# Tahmin
##########################

# 150 birimlik tv harcaması olsa ne kadar satış olur?
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

# 500 birimlik tv harcaması olsa ne kadar satış olur?
reg_model.intercept_[0] + reg_model.coef_[0][0] * 500

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")
g.set_title(
    f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

##########################
# Tahmin Başarısı
##########################

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# R-KARE
# bağımsız değişkenlerin, bağımlı değişkendeki değişimi (varyansı) açıklama yüzdesidir
reg_model.score(X, y)

######################################################
# * Multiple Linear Regression
######################################################

df = pd.read_csv("datasets/advertising.csv")
X = df.drop('sales', axis=1)
y = df[["sales"]]

##########################
# Model
##########################

# veri setini eğitim ve test olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_

##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40


# El ile:
# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002
2.90 + 30 * 0.04 + 10 * 0.17 + 40 * 0.002

# Fonksiyonel:
# 3 ayrı liste, yani bağımsız 3 değişkeni temsil etmekte
yeni_veri = [[30], [10], [40]]

yeni_veri = pd.DataFrame(yeni_veri).T

# gelen yeni veriden (bağımsız değişkenlerden) tahmin edilen bağımlı değişken değeri (bu örnek için 'sales')
reg_model.predict(yeni_veri)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# TRAIN RMSE
# eğitim bağımlı değişkenlerini kullanarak bağımsız değişkeni tahmin ediyoruz
y_pred = reg_model.predict(X_train)
# eğitim bağımlı değişkenleri ile tahmin edilen bağımlı değişkenleri karşılaştırıyoruz
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN RKARE
reg_model.score(X_train, y_train)

# TEST RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# TEST  RKARE
reg_model.score(X_test, y_test)

# 10 Katlı CV RMSE
np.mean( # 10 adet RMSE'nin ortalamasını aldık
    np.sqrt( # karekökünü aldık. 10 validasyon işleminin hataları
        -cross_val_score(reg_model, X, y, cv=10,
                         scoring="neg_mean_squared_error") # Cross-Validation: veri setini 10 parçaya böldük. Önce 9 parça ile model kurup 1'i ile test ediyor, sonraki iterasyonda diğer 9 parça ve 1 ...
    )
)


######################################################
# * Simple Linear Regression with Gradient (Batch) Descent from Scratch
######################################################

# Cost function
def cost_function(Y, b, w, X):
    m = len(Y)  # gözlem sayısı
    sse = 0  # toplam hata
    
    # butun gozlem birimlerini gez:
    for i in range(0, m):
        y_hat = b + w * X[i] # linear regression denklemi, tahmin
        y = Y[i] # bağımlı değişken
        sse += (y_hat - y) ** 2 # toplam hata, y - y_hat
    
    mse = sse / m # MSE
    return mse

# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y) # gözlem sayısı

    # kısmi türevler başta 0 olarak atanıyor
    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m): # tüm gözlem birimlerinde gez

        y_hat = b + w * X[i] # tahmin edilen değerleri hesapla

        y = Y[i] # bağımlı değişkeni seç

        b_deriv_sum += (y_hat - y) # türevin sonucu (gradyan)

        w_deriv_sum += (y_hat - y) * X[i] # türev (y_hat - y), ilgili gözlem birimindeki bağımsız değişken ile çarpıldı

    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                            cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w

    cost_history = []

    for i in range(num_iters):

        b, w = update_weights(Y, b, w, X, learning_rate)

        mse = cost_function(Y, b, w, X)

        cost_history.append(mse)

        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(
                i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(
        num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("datasets/advertising.csv")


X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

train(Y, initial_b, initial_w, X, learning_rate, num_iters)
