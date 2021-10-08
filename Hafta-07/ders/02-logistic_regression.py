######################################################
# Diabetes Predciction with Logistic Regression
######################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



######################################################
# Exploratory Data Analysis
######################################################

df = pd.read_csv("datasets/diabetes.csv")

##########################
# Target'ın Analizi
##########################

# Tüm sayısal değişkenlerin özet istatistikleri:
df.describe().T

# Hedef değişkenin sınıfları ve frekansları:
df["Outcome"].value_counts()

# Frekanslar görsel olarak
sns.countplot(x="Outcome", data=df)
plt.show()

# Hedef değişkenin sınıf oranları:
100 * df["Outcome"].value_counts() / len(df)


##########################
# Feature'ların Analizi
##########################

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()


def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show()

# Tüm değişkenlere direk uygulayabiliriz:
for col in df.columns:
    plot_numerical_col(df, col)

# Outcome'ı dışarıda bırakmak istersek:
cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    plot_numerical_col(df, col)

##########################
# Target vs Features
##########################

df.groupby("Outcome").agg({"Pregnancies": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in cols:
    target_summary_with_num(df, "Outcome", col)


######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################


# Eksik değer incelemesi:
df.isnull().sum()

# Aykırı değer incelemesi:
for col in cols:
    print(col, check_outlier(df, col))


# Aykırı değerlerin eşik değerler ile değiştirilmesi:
replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


######################################################
# Model & Prediction
######################################################

# Bağımlı ve bağımsız değişkelerin seçilmesi:
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Model:
log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_


# Tahmin
y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]



######################################################
# Model Evaluation
######################################################


# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)


# Başarı skorları:
print(classification_report(y, y_pred))


# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

# 0.8393955


######################################################
# Model Validation: Holdout
######################################################


# Holdout Yöntemi

# Veri setinin train-test olarak ayrılması:
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)


# Modelin train setine kurulması:
log_model = LogisticRegression().fit(X_train, y_train)

# Test setinin modele sorulması:
y_pred = log_model.predict(X_test)

# AUC Score için y_prob (1. sınıfa ait olma olasılıkları)
y_prob = log_model.predict_proba(X_test)[:, 1]


# Classification report
print(classification_report(y_test, y_pred))


# ROC Curve
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

roc_auc_score(y_test, y_prob)



######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)


cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])



cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327


######################################################
# Prediction for A New Observation
#####################################################

X.columns

random_user = X.sample(1, random_state=44)

log_model.predict(random_user)





























