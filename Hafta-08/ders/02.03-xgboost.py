################################################
# XGBoost
################################################

import warnings
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
from xgboost import XGBClassifier

# !pip install xgboost

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("datasets/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001], # büyüme şiddeti
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000], # ağaç sayısı, iterasyon sayısı..
                  "colsample_bytree": [0.5, 0.7, 1] # yüzdelik olarak kaç gözlem bulunsun
                  }

xgboost_best_grid = GridSearchCV(xgboost_model,
                                 xgboost_params,
                                 cv=5,
                                 n_jobs=-1, verbose=True).fit(X, y)

xgboost_best_grid.best_score_

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_,
                                         random_state=17).fit(X, y)


cv_results = cross_validate(xgboost_final,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])



cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
