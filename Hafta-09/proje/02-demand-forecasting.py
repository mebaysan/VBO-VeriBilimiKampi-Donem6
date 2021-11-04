#####################################################
# * Demand Forecasting
#####################################################

# Store Item Demand Forecasting Challenge (Kaggle yarışması)
# https://www.kaggle.com/c/demand-forecasting-kernels-only

# Farklı store için 3 aylık item-level sales tahmini.
# 5 yıllık bir veri setinde 10 farklı mağaza ve 50 farklı item var.
# Buna göre mağaza-item kırılımında 3 ay sonrasının tahminlerini vermemiz gerekiyor.

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


########################
# * Loading the data
########################

train = pd.read_csv('datasets/demand_forecasting/train.csv', parse_dates=['date']) # 2018 ilk 3 ay öncesindeki aylar
test = pd.read_csv('datasets/demand_forecasting/test.csv', parse_dates=['date']) # 2018 ilk 3 ay
sample_sub = pd.read_csv('datasets/demand_forecasting/sample_submission.csv')
df = pd.concat([train, test], sort=False)

#####################################################
# * EDA
#####################################################

df["date"].min(), df["date"].max()
check_df(train)
check_df(test)

check_df(df)

# Satış dağılımı nasıl?
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

# Kaç store var?
df[["store"]].nunique()

# Kaç item var?
df[["item"]].nunique()

# Her store'da eşit sayıda mı eşsiz item var?
df.groupby(["store"])["item"].nunique()

# Peki her store'da eşit sayıda mı sales var?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# mağaza-item kırılımında satış istatistikleri
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

#####################################################
# * FEATURE ENGINEERING
#####################################################


########################
# * Date Features
########################


def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df


df = create_date_features(df)

check_df(df)  

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


########################
# * Random Noise
########################
# veriye kendimiz rasgele gürültü ekleyeceğim. Rassalığı oluşturabilmek adına

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


########################
# * Lag/Shifted Features
########################
# gecikme feature'ları türeteceğim.
# geçmiş gerçek değerleri türetecğiz yani
# buradaki temel mantık, zaman serisinin en çok kendisinden önceki değerden etkileniyor olduğunu kabul etmemizdir;
# ve uğraştığımız set tek değişkenli bir veri seti olmadığından, kendimizce özellikler türetiyoruz

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True) # önemli bir adım.

check_df(df)

# satışın ilk 10 gözlemine bakalım:
df["sales"].head(10)

# Birinci gecikme
df["sales"].shift(1).values[0:10] # her gözlemin 1 öncesindeki gözlemi verir. Doğal olarak 1. sıradaki gözlem için arkasında bir gözlem olmadığından nan gelecektir.

# İkinci gecikme
df["sales"].shift(2).values[0:10]

# Üçüncü gecikme
df["sales"].shift(3).values[0:10]

# Anlaşılır olması için hepsini bir arada gözlemleyelim
pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})


df.groupby(["store", "item"])['sales'].head()

df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1)) # transform sayesinde direkt olarak hesapladığımız değeri atayabiliyoruz

def lag_features(dataframe, lags): # mevsimsellik etkisi yakalamaya çalışıyorum. 
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

check_df(df)

df[df['sales'].isnull()]

########################
# * Rolling Mean Features
########################
# hareketli ortalama feature'ları

# window = gözlemden kaç adım geriye (gözlemin kendisi dahil) gidilecek
# gözlem, kendisi dahil {window} adım kadar geri gider ve gözlemleri toplar, ortalamasını alır, döner.
# gözlemin kendisi dahil, {window} önceki değerleri toplar ve {window} 'a böler (ortalamalarını alır).
# gözlemden önce {window} kadar gözlem yoksa nan döner.

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

# * hareketli ortalamalar üretilirken öncelikle 1 shift almak gerekir. Gerçek önceki değerleri üretmek istediğimizde gözlemin kendisini ortalama hesabına katmamalıyız. Bu sebeple öncelikle shift aldık sonra hareketli ortalama hesapladık.
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546]) # 365 gün ve 546 gün öncesinden hareketli ortalamaları alıyoruz
df.tail()


########################
# * Exponentially Weighted Mean Features
########################
# ağırlıklı hareketli ortalama
# ewm içerisindeki alpha değeri 0 ile 1 arasında değer alır. 1'e ne kadar yakınsa o kadar son değerlere ağırlık verir. 

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm01": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

check_df(df)


########################
# * One-Hot Encoding
########################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])


########################
# * Converting sales to log(1+sales)
########################

df['sales'] = np.log1p(df["sales"].values) # bağımlı değişkenin logaritmasını alıyorum. GBM temelli bir algoritma kullanacağımdan dolayı optimizasyon süresini kısaltmak için logaritmasını alıyorum.
check_df(df)


#####################################################
# * Model
#####################################################

########################
# * Custom Cost Function
########################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


########################
# * Time-Based Validation Sets
########################

test

# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]

val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


########################
# LightGBM Model
########################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 20000,
              'early_stopping_rounds': 200
              }


# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))


########################
# Değişken önem düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)
plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()

########################
# Final Model
########################



train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}


# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = model.predict(X_test, num_iteration=model.best_iteration)


submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv('submission_demand.csv', index=False)
submission_df.head(20)


# Pazartesi hedef salıyı tahmin etmek. geçmiş günleri kullanarak tahmine ettim.
# Bugün pazartesi hedef çarşambayı tahmin etmek? Salıyı tahmin et.
# Bu tahminleri gerçek değer olarak kabul edip çarşambayı bunun üzerinden tahmin et.
# Direk 2. periyodu tahmin et.