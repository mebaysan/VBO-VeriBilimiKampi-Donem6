######################################################
# Automated Machine Learning
######################################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from helpers.eda import *
from helpers.data_prep import *

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

######################################################
# Data Preprocessing & Feature Engineering
######################################################


df = pd.read_csv("datasets/titanic.csv")

def titanic_data_prep(dataframe):
    # 1. Feature Engineering (Değişken Mühendisliği)
    dataframe.columns = [col.upper() for col in dataframe.columns]
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype('int')
    dataframe["NEW_NAME_COUNT"] = dataframe["NAME"].str.len()
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["NAME"].apply(lambda x: len(str(x).split(" ")))
    dataframe["NEW_NAME_DR"] = dataframe["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    dataframe['NEW_TITLE'] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # 2. Outliers (Aykırı Değerler)
    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # 3. Missing Values (Eksik Değerler)
    dataframe.drop("CABIN", inplace=True, axis=1)
    remove_cols = ["TICKET", "NAME"]
    dataframe.drop(remove_cols, inplace=True, axis=1)
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
            (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x,
                                axis=0)

    # 4. Label Encoding
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
                   and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # 5. Rare Encoding
    dataframe = rare_encoder(dataframe, 0.01)

    # 6. One-Hot Encoding
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # 7. Standart Scaler
    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    return dataframe

df_prep = titanic_data_prep(df)

y = df_prep["SURVIVED"]
X = df_prep.drop(["PASSENGERID", "SURVIVED"], axis=1)


######################################################
# Modeling
######################################################

classifiers = [('LR', LogisticRegression()),
               ('KNN', KNeighborsClassifier()),
               ("SVC", SVC()),
               ("CART", DecisionTreeClassifier()),
               ("RF", RandomForestClassifier()),
               ('Adaboost', AdaBoostClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ('XGBoost', XGBClassifier()),
               ('LightGBM', LGBMClassifier()),
               # ('CatBoost', CatBoostClassifier(verbose=False))
               ]


for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X, y, cv=3, scoring=["roc_auc"])
    print(f"AUC: {round(cv_results['test_roc_auc'].mean(),4)} ({name}) ")



######################################################
# Automated Hyperparameter Optimization
######################################################


knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500, 1000]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


best_models = {}


for name, classifier, params in classifiers:
    print(f"########## {name} ##########")
    cv_results = cross_validate(classifier, X, y, cv=3, scoring=["roc_auc"])
    print(f"AUC (Before): {round(cv_results['test_roc_auc'].mean(),4)}")


    gs_best = GridSearchCV(classifier, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
    final_model = classifier.set_params(**gs_best.best_params_)

    cv_results = cross_validate(final_model, X, y, cv=3, scoring=["roc_auc"])
    print(f"AUC (After): {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model


######################################################
# Stacking & Ensemble Learning
######################################################

voting_clf = VotingClassifier(
    estimators=[('XGBoost', best_models["XGBoost"]),
                ('RF', best_models["RF"]),
                ('LightGBM', best_models["LightGBM"])],
    voting='soft')

voting_clf.fit(X, y)

cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


######################################################
# Prediction for a New Observation
######################################################

X.columns

random_user = X.sample(1, random_state=45)


voting_clf.predict(random_user)

import pickle

pickle.dump(open(voting_clf, "voting_clf.pkl", "wb"))


pickle.dump(voting_clf, open('voting_clf.pkl', 'wb'))