########################################
##########  Encoding (Label Encoding, One-Hot Encoding, Rare Encoding) ##########
########################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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


#############################################
# Label Encoding & Binary Encoding
#############################################


df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5] # neye göre 0, neye göre 1? Alfabetik sıraya göre ilk yakaladığına 0 diğerine 1 der
le.inverse_transform([0, 1]) # bu metod sayesinde 0'a ne dendi ve 1'e ne dendiyi tutabiliriz


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


df = load()


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]


len(binary_cols)


for col in binary_cols:
    label_encoder(df, col)

df.head()


df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)

df[binary_cols].head()


# * Önemli bir soru: neden nunique != len(unique)? unique metodu eksik değeri de eşsiz değer olarak görür
df = load()
df["Embarked"].nunique()
len(df["Embarked"].unique())


#############################################
# One-Hot Encoding
#############################################
# * Genelde 2'den fazla sınıflı değişkenler için kullanırız fakat pratik yol ararsak tüm hepsi için yapabiliriz. Tüm kategorik değişkenler için yapacaksak drop_first argümanı kullanılmalıdır.
df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head() # dummy_na ile eksik değerler için de değişken oluşturur
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head() # drop_first ile dummy değişken tuzağından kurtulmak için ilk sınıfı düşürürüz. Bu örnekte C düşer. Alfabetik olarak baktığını unutmamalıyız.

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = load()
# * Problem: Sayısal görünümlü kategorik değişkenlerimiz var. Bu örnek için PClass ve SibSp. Yine bu senaryoya özel: sınıf sayısı 10 veya daha küçük ya da 2 veya daha büyük olan değişkenleri al dedik
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols).head()
one_hot_encoder(df, ohe_cols, drop_first=True).head()


#############################################
# Rare Encoding
#############################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()


cat_cols, num_cols, cat_but_car = grab_col_names(df)

from helpers.eda import cat_summary

for col in cat_cols:
    cat_summary(df, col)


###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################


df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# 1. Sınıf Frekansı
# 2. Sınıf Oranı
# 3. Sınıfların target açısından değerlendirilmesi

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", cat_cols)


#############################################
# 3. Rare encoder'ın yazılması.
#############################################




temp_df = df.copy()

temp_df["ORGANIZATION_TYPE"].unique()
len(temp_df["ORGANIZATION_TYPE"].unique())
temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)
tmp = temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)

rare_labels = tmp[tmp < 0.01].index
len(rare_labels), rare_labels



temp_df["ORGANIZATION_TYPE"] = np.where(temp_df["ORGANIZATION_TYPE"].isin(rare_labels), 'Rare',
                                        temp_df["ORGANIZATION_TYPE"])

len(temp_df["ORGANIZATION_TYPE"].unique())
temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)
rare_analyser(df, "TARGET", cat_cols)


