#######################################
# * Hitters ML Pipeline - M.Enes Baysan
# * Bu versiyonda NaN değerler drop edilmiştir ve V1 ile V2'den elde edilen tecrübelere göre RandomForests, GBM ve LightGBM Kullanılmıştır
# * Bu dosyada; RandomizedSearchCV yardımıyla en optimal hiperaparametreleri bulmaya odaklanılmıştır
#######################################

import pandas as pd
import numpy as np
from helpers import data_prep, eda, model_evaluation
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, RandomizedSearchCV


warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("datasets/hitters.csv")
df.head()

#######################################
# * Data Preprocessing
#######################################
df.isna().sum()
df = df.dropna()
cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)


for col in cat_cols:
    eda.cat_summary(df, col)

eda.check_df(df)

df.head()

# * 1 -) Eski numerik değişkenlerden outlier'ları temizleyip yeni outlier'sız değişkenler oluşturdum
# * 2-) Yeni outlier'sız değişkenlerden kategorize edilmiş değişkenler oluşturdum
for i in num_cols:
    if data_prep.check_outlier(df, i, 0.25, 0.75):
        min_th, max_th = data_prep.outlier_thresholds(df, i, 0.25, 0.75)
        df[f'NEW_{i}'] = df[i]
        df.loc[(df[f'NEW_{i}'] < min_th), f'NEW_{i}'] = min_th
        df.loc[(df[f'NEW_{i}'] > max_th), f'NEW_{i}'] = max_th
        df[f'CAT_{i}'] = pd.qcut(df[f'NEW_{i}'], 4, ['D', 'C', 'B', 'A'])
    else:
        df[f'NEW_{i}'] = df[i]
        df[f'CAT_{i}'] = pd.qcut(df[f'NEW_{i}'], 4, ['D', 'C', 'B', 'A'])

df.head()

drop_cols = [i for i in df.columns if 'NEW_' in i]

df = df.drop(drop_cols, axis=1)

cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

#rob_scaler = RobustScaler()
#df[num_cols] = rob_scaler.fit_transform(df[num_cols])

df = pd.get_dummies(df, drop_first=True)

######################################################
# * Base Models
######################################################
y = df["Salary"]
X = df.drop(["Salary"], axis=1)

models = [
    # V1 ve V2'den elde ettiğim tecrübeler sonucu seçtiğim modeller
    ('RF', RandomForestRegressor()),
    ('GBM', GradientBoostingRegressor()),
    ("LightGBM", LGBMRegressor()),
]

results_dict = {'Model': [], 'RMSE': []}

for name, regressor in models:
    # * bu problemde maaş tahmini yaptığımız için hata metriğimiz mean squarred error oldu
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y,
                   cv=10, scoring="neg_mean_squared_error")))
    # print(f"RMSE: {round(rmse, 4)} ({name}) ")
    results_dict['Model'].append(name)
    results_dict['RMSE'].append(round(rmse, 4))

results_df = pd.DataFrame(results_dict).sort_values('RMSE', ascending=True)

######################################################
# * Automated Hyperparameter Searching
######################################################

# * Modellere uygulanabilecek parametre değerlerini (aralıklarını) veriyorum
rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

gbm_random_params = {'max_depth': np.random.randint(5, 50, 10),
                     "min_samples_split":  np.random.randint(2, 50, 20)}

lightgbm_random_params = {"learning_rate": np.random.random(20),
                          "n_estimators": np.random.randint(100, 2000, 20),
                          "colsample_bytree": np.random.random(20)}

# * for loopta kullanabilmek için model adı, model ve parametrelerinin bulunduğu tuple
regressors = [("GBM", GradientBoostingRegressor(), gbm_random_params),
              ("RF", RandomForestRegressor(), rf_random_params),
              ('LightGBM', LGBMRegressor(), lightgbm_random_params),
              ]

# * sonuçları dataframe olarak almak için bir dict oluşturuyorum. DataFrame'e çevireceğim
random_optimized_results_dict = {
    'Name': [], 'RMSE (Before)': [], 'RMSE (After)': [], 'Best Params': []}

for name, regressor, params in regressors:
    random_optimized_results_dict['Name'].append(name) # * hangi model

    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y,cv=10, scoring="neg_mean_squared_error"))) # * looptaki modele parametre optimizasyonu yapılmadan boş modelde rmse
    random_optimized_results_dict['RMSE (Before)'].append(rmse)

    best_random_model = RandomizedSearchCV(estimator=regressor, param_distributions=params, cv=5, n_iter=100, verbose=False).fit(X, y) # * RandomizedSeachCV ile rastgele en iyi parametrelerin bulunması
    random_optimized_results_dict['Best Params'].append(str(best_random_model.best_params_)) # * rastgele bulunan en iyi parametreler

    final_model = regressor.set_params(**best_random_model.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y,cv=10, scoring="neg_mean_squared_error"))) # * rastgele en iyi parametrelerle kurulan modelden elde edilen rmse
    random_optimized_results_dict['RMSE (After)'].append(rmse) # * rastgele en iyi parametreler sonrası rmse
    

random_optimized_results_df = pd.DataFrame(random_optimized_results_dict)

for i in random_optimized_results_df['Best Params']:
    print(i)


######################################################
# * Automated Hyperparameter Optimization
######################################################

rf_params = {"max_depth": [20,23,30,50,35,15],
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": [10,20,25,30,35,40],
                    "n_estimators": [650,633,615,600,700,750]}

gbm_params = {'max_depth': [8,16,32,10,12],
                     "min_samples_split":  [20,25,30,38,42,50,60]}

lightgbm_params = {"learning_rate": [0.029732255062551505,0.029732255062551505 + 0.0500, 0.029732255062551505 - 0.0500],
                          "n_estimators": [495,450,550,520,490,510],
                          "colsample_bytree": [0.9895650118857221, 0.9895650118857221 - 0.05, 0.9895650118857221 - 0.02]}

regressors = [("GBM", GradientBoostingRegressor(), gbm_params),
              ("RF", RandomForestRegressor(), rf_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params),
              ]

best_models = {}
optimized_results_dict = {
    'Name': [], 'RMSE (Before)': [], 'RMSE (After)': [], 'Best Params': []}

for name, regressor, params in regressors:
    optimized_results_dict['Name'].append(name)

    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y,cv=10, scoring="neg_mean_squared_error")))
    optimized_results_dict['RMSE (Before)'].append(rmse)

    best_model = GridSearchCV(estimator=regressor, param_grid=params, cv=5, verbose=False).fit(X, y)
    optimized_results_dict['Best Params'].append(str(best_model.best_params_))
    final_model = regressor.set_params(**best_model.best_params_)
    
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y,cv=10, scoring="neg_mean_squared_error")))
    optimized_results_dict['RMSE (After)'].append(rmse)

    best_models[name] = final_model

optimized_results_df = pd.DataFrame(optimized_results_dict)
random_optimized_results_df

######################################################
# * Stacking & Ensemble Learning
######################################################

voting_reg = VotingRegressor(estimators=[('GBM', best_models["GBM"]),
                                         ('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])
                                         ],)

voting_reg.fit(X, y)


np.mean(np.sqrt(-cross_val_score(voting_reg, X, y,
        cv=10, scoring="neg_mean_squared_error")))

######################################################
# * Prediction for a New Observation
######################################################

X.columns
random_user = X.sample(1, random_state=45)
voting_reg.predict(random_user)
