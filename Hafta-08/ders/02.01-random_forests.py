################################################
# Random Forests
################################################
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate


pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("datasets/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)


################################################
# Random Forests
################################################


rf_model = RandomForestClassifier(random_state=17) # model nesnesi oluşturuyoruz

# hyperparameter optimizasyonu yapmak için hiperparametreleri yazıyorum
rf_params = {"max_depth": [5, 8, None], # max ağaç derinliği
             "max_features": [3, 5, 7, "auto"], # bölünme işlemi yapılırken göz önünde bulundurulacak olan değişken sayısı
             "min_samples_split": [2, 5, 8, 15, 20], # bir node'u dala ayırmak için gerekli minimum gözlem sayısı
             "n_estimators": [100, 200, 500]} # ağaç sayısı, kolektif (topluluk) öğrenme metodu olduğundan kaç adet ağaç olmasını istiyoruz

rf_best_grid = GridSearchCV(rf_model, # hangi model
                            rf_params, # hangi parametreler
                            cv=5, # kaç katlı çaprazlama
                            n_jobs=-1, # tüm işlemcileri kullan
                            verbose=True).fit(X, y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_, # en iyi hiperparametreleri modele set ediyorum
                               random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, # final modelin cross validation hatası
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
# 0.7344326725905673
cv_results['test_f1'].mean()
#  0.5701221536747852
cv_results['test_roc_auc'].mean()
# 0.7710925925925926