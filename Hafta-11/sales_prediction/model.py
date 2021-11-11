import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


df = pd.read_csv("/Users/mvahit/Documents/DSMLBC5/datasets/advertising.csv")
X = df.drop('sales', axis=1)
y = df[["sales"]]

reg_model = LinearRegression()
reg_model.fit(X, y)

y_pred = reg_model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE:", rmse)

pickle.dump(reg_model, open('regression_model.pkl', 'wb'))
print("model created")
# model = pickle.load(open('regression_model.pkl','rb'))
# print(model.predict())
