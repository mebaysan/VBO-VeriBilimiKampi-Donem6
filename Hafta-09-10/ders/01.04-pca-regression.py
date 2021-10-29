
################################
# * BONUS
################################
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder


################################
# * BONUS: Principal Component Regression
################################
df = pd.read_csv("datasets/Hitters.csv")

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


pca_fit.shape


df = pd.read_csv("datasets/Hitters.csv")
df.shape

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]).head()

df[others].head()


final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                      df[others]], axis=1)

final_df.head()


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)


final_df.dropna(inplace=True)


y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))

y.mean()


cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))

