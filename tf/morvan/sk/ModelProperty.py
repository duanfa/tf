from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression(   )
model.fit(data_X,data_y)

# y = ax+b
print(model.coef_)  # print a
print(model.intercept_)
print(model.get_params()) 
print(model.score(data_X,data_y)) # 