######################################################
# Diabetes Prediction with Logistic Regression
######################################################


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split
from helpers.eda import *
from helpers.data_prep import *


##########################
# Logistic Regression Sürecini Anlamak
##########################


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# bias
b = -0.03

# weight
w = 0.001

# bağımlı değişken gerçek değerleri
y = pd.Series([1, 0, 1, 0, 0, 1, 1, 1, 1])

# bağımsız değişken değerleri
x = pd.Series([85, 20, 56, 0, 1, 30, 8, 28, 100])


# z
z = b + np.dot(x, w)

# sigmoid'in üreteceği 1 sınıfına ait olma olasılıkları
y_prob = sigmoid(z)

# belirli bir eşik değerine göre bu olasılıkların 1-0 sınıf tahminlerine dönüştürülmesi.
df = pd.DataFrame()

def classification_threshold(p, th=0.50):
    if p >= th:
        return 1
    else:
        return 0

for i, p in enumerate(y_prob):
    df.loc[i, "y_pred"] = classification_threshold(p)

df
df["y"] = y
df["x"] = x
df["z"] = z
df["y_prob"] = y_prob

df = df[["y", "x", "z", "y_prob", "y_pred"]]
df

accuracy_score(df["y"], df["y_pred"])

##########################
# Manuel Logloss
##########################

# log loss
yi = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
y_prob = [0.80, 0.48, 0.30, 0.45, 0.55, 0.70, 0.42, 0.35, 0.60, 0.70]

# birinci gözlem için 1 ve 0.80 değerlerine göre logloss
-1*(np.log10(0.80))
# 0.096

# üçüncü gözlem için 0 ve 0.30 değerlerine göre logloss
-1 * np.log10(1 - 0.30)
# 0.15

# logloss
log_loss_sum = 0

for i, y in enumerate(yi):
    if y == 1:
        log_loss_sum += -1 * (np.log10(y_prob[i]))
    else:
        log_loss_sum += -1 * np.log10(1 - y_prob[i])


log_loss_sum/len(yi)




##########################
# Keşifçi Veri Analizi
##########################

df = pd.read_csv("datasets/diabetes.csv")

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_summary(df, "Outcome")

for col in num_cols:
    num_summary(df, col, plot=True)

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


##########################
# Data Preprocessing (Veri Ön İşleme)
##########################

# eksik değer incelemesi
missing_values_table(df)

# aykırı değer incelemesi
for col in num_cols:
    print(col, check_outlier(df, col))


grab_outliers(df, "Insulin")

# aykırı değerlerin silinmesi
replace_with_thresholds(df, "Insulin")


for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])


check_df(df)




##########################
# Logistic Regression
##########################

##########################
# Model
##########################


y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

log_model = LogisticRegression().fit(X_train, y_train)
log_model.intercept_
log_model.coef_



##########################
# Tahmin
##########################

# tahmin'lerin oluşturulması ve kaydedilmesi
y_pred = log_model.predict(X_train)
y_pred[0:10]
y_train[0:10]

# sınıf olasılıkları
log_model.predict_proba(X_train)[0:10]

# 1. sınıfa ait olma olasılıkları:
y_prob = log_model.predict_proba(X_train)[:, 1]

##########################
# Başarı Değerlendirme
##########################


# Train Accuracy
y_pred = log_model.predict(X_train)
accuracy_score(y_train, y_pred)


# Test
# AUC Score için y_prob
y_prob = log_model.predict_proba(X_test)[:, 1]

# Diğer metrikler için y_pred
y_pred = log_model.predict(X_test)


# CONFUSION MATRIX
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y_test, y_pred)


# ACCURACY
accuracy_score(y_test, y_pred)

# PRECISION
precision_score(y_test, y_pred)

# RECALL
recall_score(y_test, y_pred)

# F1
f1_score(y_test, y_pred)

# ROC CURVE
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

# Classification report
print(classification_report(y_test, y_pred))