########################################
##########  Feature Scaling (Özellik Ölçeklendirme) ##########
########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

# * Hafta 2'de yazdığımız fonksiyon
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.describe().T

###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

rs = RobustScaler() # Robust: aykırılığa dayanıklıdır.
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

###################
# Log: Logaritmik dönüşüm.
###################

# not: - degerler varsa logaritma alınamayacağı için bu durum göz önünde bulundurulmalı.
df["Age_log"] = np.log(df["Age"])
df.describe().T

age_cols = [col for col in df.columns if "Age" in col]


for col in age_cols:
    num_summary(df, col, plot=True)


###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
###################

bins = [0, 18, 23, 30, 40, int(df["Age"].max())]
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(int(df["Age"].max()))]

# age'i bölelim:
df["age_cat"] = pd.cut(df["Age"], bins, labels=mylabels)
df.head()


# qcut
df["Age_qcut"] = pd.qcut(df['Age'], 5)

df["Age_qcut"].value_counts()
df["age_cat"].value_counts()

