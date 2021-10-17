################################################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
################################################
"""
RandomSearchCV: hiperparametre komninasyonlarından rastgele N adedini seçer ve dener. Daha geniş bir aralık verebiliriz. 
                GridSearchCV'de tüm kombinasyonları denedğinden, fazla hiperparametre kombinasyonu fazla işlem maliyetini beraberinde getirecektir

                Uygulanabilecek senaryo: RandomizedSearchCV sonucunda elde ettiğimiz N adet rastgele kombinasyonu GridSearchCV'e verebiliriz.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV


pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("datasets/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

################################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
################################


rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}


rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)

# En iyi hiperparametre değerleri:
rf_random.best_params_

# En iyi skor
rf_random.best_score_


rf_random_final = rf_model.set_params(**rf_random.best_params_,
                                      random_state=17).fit(X, y)


cv_results = cross_validate(rf_random_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

