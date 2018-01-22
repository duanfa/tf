import numpy as np

x_datas = np.linspace(-1,1,300)[:,np.newaxis]
in_size,out_size=1,10

Weights = np.random.normal([in_size,out_size])

result = np.multiply(x_datas,Weights)
result2 = np.multiply(Weights,x_datas)

print(Weights.shape)
print(x_datas.shape)
print(result.shape)
print(result2.shape)