#######################################
# * Hitters ML Pipeline - M.Enes Baysan
# * Bu versiyonda NaN değerler drop edilmiştir
#######################################

# !pip install xgboost
# !pip install lightgbm
# !pip install catboost

import pandas as pd
import numpy as np
from helpers import data_prep, eda, model_evaluation
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate


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
        min_th, max_th = data_prep.outlier_thresholds(df,i,0.25,0.75)
        df[f'NEW_{i}'] = df[i]
        df.loc[(df[f'NEW_{i}'] < min_th), f'NEW_{i}'] = min_th
        df.loc[(df[f'NEW_{i}'] > max_th), f'NEW_{i}'] = max_th
        df[f'CAT_{i}'] = pd.qcut(df[f'NEW_{i}'],4,['D','C','B','A'])
    else:
        df[f'NEW_{i}'] = df[i]
        df[f'CAT_{i}'] = pd.qcut(df[f'NEW_{i}'],4,['D','C','B','A'])

df.head()

drop_cols = [i for i in df.columns if 'NEW_' in i]

df = df.drop(drop_cols,axis=1)

cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

#rob_scaler = RobustScaler()
#df[num_cols] = rob_scaler.fit_transform(df[num_cols])

df = pd.get_dummies(df,drop_first=True)

######################################################
# * Base Models
######################################################
y = df["Salary"]
X = df.drop(["Salary"], axis=1)

models = [
          ############# Denedim ve İyi Performans Alamadığım Modeller
          ('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('SVR', SVR()),
          ("CatBoost", CatBoostRegressor(verbose=False)),
          ############# Denedim ve İyi Performans Aldığım Modeller
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ]

results_dict = {'Model':[],'RMSE':[]}

for name, regressor in models:
    # * bu problemde maaş tahmini yaptığımız için hata metriğimiz mean squarred error oldu
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y,cv=10, scoring="neg_mean_squared_error")))
    # print(f"RMSE: {round(rmse, 4)} ({name}) ")
    results_dict['Model'].append(name)
    results_dict['RMSE'].append(round(rmse,4))

results_df = pd.DataFrame(results_dict).sort_values('RMSE',ascending=True)

######################################################
# * Automated Hyperparameter Optimization
######################################################



rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500, 1000]}

gbm_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200, 300, 500],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                   "n_estimators": [300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}

# catboost_params = {"iterations": [200, 500],
#                    "learning_rate": [0.01, 0.1],
#                    "depth": [3, 6]
#                    }

regressors = [("GBM", GradientBoostingRegressor(), gbm_params),
              ("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params),
              #('CatBoost', CatBoostRegressor(), catboost_params),
              ]

best_models = {}
optimized_results_dict = {'Name':[],'RMSE (Before)':[],'RMSE (After)':[],'Best Params':[]}

for name, regressor, params in regressors:
    #print(f"########## {name} ##########")
    optimized_results_dict['Name'].append(name)
    
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    #print(f"RMSE: {round(rmse, 4)} ({name}) ")
    optimized_results_dict['RMSE (Before)'].append(rmse)

    gs_best = GridSearchCV(regressor, params, cv=3, verbose=False).fit(X, y)
    optimized_results_dict['Best Params'].append(str(gs_best.best_params_))
    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    optimized_results_dict['RMSE (After)'].append(rmse)
    
    #print(f"RMSE (After): {round(rmse, 4)} ({name}) ")
    #print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

optimized_results_df = pd.DataFrame(optimized_results_dict)

######################################################
# * Stacking & Ensemble Learning
######################################################

voting_reg = VotingRegressor(estimators=[('GBM', best_models["GBM"]),
                                         ('RF', best_models["RF"])],)

voting_reg.fit(X, y)


np.mean(np.sqrt(-cross_val_score(voting_reg, X, y,
        cv=10, scoring="neg_mean_squared_error")))

######################################################
# * Prediction for a New Observation
######################################################

X.columns
random_user = X.sample(1, random_state=45)
voting_reg.predict(random_user)
