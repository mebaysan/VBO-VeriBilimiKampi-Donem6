########################################
########## Outliers ##########
########################################

from os import remove
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data


dff = load_application_train()
dff.head()


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()


########################################
### Aykırı değerleri görselleştirme ####
########################################
# 'Age' değişkeni için boxplot çizdik. Bıyıkların dışında kalan noktalar bize aykırı değerleri verir
sns.boxplot(df['Age'])
plt.show()


########################################
### Aykırı değerleri nasıl yakalarız? (IQR Yöntemi) ####
########################################

q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)
iqr = q3 - q1
up_limit = q3 + 1.5 * iqr
low_limit = q1 - 1.5 * iqr

# Age değişkeni aykırı olan gözlemler
df[(df['Age'] < low_limit) | (df['Age'] > up_limit)]
df[(df['Age'] < low_limit) | (df['Age'] > up_limit)]['Age']  # aykırı değerler
df[(df['Age'] < low_limit) | (df['Age'] > up_limit)].index  # index bilgileri

df[(df['Age'] < low_limit) | (df['Age'] > up_limit)].any(
    axis=None)  # Herhangi bir aykırı değeri var mı?


# * Aykırı değerler için sınırları veren fonksiyon
def outlier_thresholds(dataframe, column_name, q1=0.25, q3=0.75):
    """
    Değişkenin IQR yöntemi için alt ve üst sınırlarını verir
    """
    q1 = dataframe[column_name].quantile(0.25)
    q3 = dataframe[column_name].quantile(0.75)
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    up_limit = q3 + 1.5 * iqr
    return low_limit, up_limit


outlier_thresholds(df, 'Age')
outlier_thresholds(df, 'Fare')

low, up = outlier_thresholds(df, 'Age')
df[(df['Age'] < low) | (df['Age'] > up)]

# * Değişekende hiç aykırı değer var mı?
def check_outlier(dataframe, column_name):
    """
    Değişkende herhangi bir aykırı değer var mı?
    """
    low, up = outlier_thresholds(dataframe, column_name)
    if dataframe[(dataframe[column_name] < low) | (dataframe[column_name] > up)].any(axis=None):
        return True
    else:
        return False

print("Yes there are outliers" if check_outlier(df,'Age') else "No there aren't outliers")

# * Onlarca değişkenimiz olsaydı ne yapardık? Yazdığımız grab_col_names fonksiyonu sayesinde hızlıca kolonları belirlerdik
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
dff = load_application_train()
cat_cols, num_cols, cat_but_car = grab_col_names(dff)

for col in num_cols:
    print(col, check_outlier(dff,col))


# * Aykırı Değerlerin Kendilerine Erişmek
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")
grab_outliers(df, "Age", True)



########################################
### Aykırı değerler problemini çözme ####
########################################

# * 1. yöntem; aykırı değere sahip gözlemleri veri setinden çıkarabiliriz
df = load()
check_outlier(df,'Age')
low, up = outlier_thresholds(df,'Age')
df[~((df['Age'] < low) | (df['Age'] > up))] # low ve up limitlerinin dışında Age değeri olan gözlemler hariç olan gözlemler

df[~df.index.isin(grab_outliers(df,'Age',True))] # grab_outliers adında kendi yazdığımız fonksiyonu kullanarak da filtreleme yapabiliriz

# Aykırı değerleri silen fonksiyon
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

df.shape # silmeden önce
remove_outlier(df,'Age').shape # sildikten sonra

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    new_df = remove_outlier(df,col)

# * 2. Yöntem: Baskılama yöntemi (re-assignment with thresholds)
df = load()
low, up = outlier_thresholds(df,'Fare', 0.05, 0.95)
df.loc[(df['Fare'] < low) | (df['Fare'] > up),'Fare']
df.loc[df['Fare'] > up,'Fare'] = up # üst sınırdan büyük olan değerlere üst sınır atandı
df.loc[df['Fare'] < low,'Fare'] = low # alt sınırdan düşük olan değerlere alt sınır atandı


# * alt ve üst limitlere göre baskılama methodu
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



########################################
### Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor ####
########################################
from sklearn.neighbors import LocalOutlierFactor

df = sns.load_dataset('diamonds')
df = df.select_dtypes(['float64','int64'])
df = df.dropna()

clf = LocalOutlierFactor(n_neighbors=20) # 20 gözlem birimi komşuluğuna bakılsın
clf.fit_predict(df) # dataframe'e uyguladık
df_scores = clf.negative_outlier_factor_ # negatif outlier factor hesapları. Bu skor ne kadar büyükse o kadar "normaldir"
# inlier değerler 1'e yakın olma eğilimindedir, outlier dediğimiz değerler 1'den küçük değerler 
df_scores[0:5]

np.sort(df_scores)[0:5] # en kötü (aykırı) 5 gözlem

scores = pd.DataFrame(np.sort(df_scores))
# elbow denilen yöntem; eğimin dikleşmeyi kestiği nokta yakalanır.
scores.plot(stacked=True,xlim=[0,20], style='.-')
plt.show()

th = np.sort(df_scores)[3] # bu örnek için eşik değer burası, plottan bakabiliriz

df[df_scores < th] # lof skoru eşik değerden düşük olan gözlemler

df.describe().T

# Gözlem sayısı az ise silinebilir
# Ya da her değişken için tek tek boxplot yöntemi ile baskılama yapılabilir










