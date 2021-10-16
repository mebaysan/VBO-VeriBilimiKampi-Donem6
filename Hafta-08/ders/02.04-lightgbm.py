################################################
# LightGBM
################################################

import warnings
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate
from lightgbm import LGBMClassifier

# !pip install lightgbm

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("datasets/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.02, 0.03, 0.1, 0.001],
               "n_estimators": [100, 250, 300, 350, 500, 1000],
               "colsample_bytree": [0.5, 0.8, 0.7, 0.6, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params,
                              cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_,
                                   random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
