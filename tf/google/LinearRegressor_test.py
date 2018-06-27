# encoding:utf-8
# https://colab.research.google.com/notebooks/mlcc/first_steps_with_tensor_flow.ipynb?hl=zh-cn#scrollTo=rhEbFCZ86cDZ
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from twisted.python.util import println

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

batch_size=10
num_epochs=5

# california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = pd.read_csv("/home/socket/tensorflow/google/onlineFiles/california_housing_train.csv", sep=",")

# 我们会将 median_house_value 调整为以千为单位，这样，模型就能够以常用范围内的学习速率较为轻松地学习这些数据。
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0

# 我们会输出关于各列的一些实用统计信息快速摘要：样本数、均值、标准偏差、最大值、最小值和各种分位数。
targets = california_housing_dataframe["median_house_value"]
# Define the input feature: total_rooms.
# println(california_housing_dataframe)
println("-------------------------------")
my_feature = california_housing_dataframe[["total_rooms"]]
features = {key:np.array(value) for key,value in dict(my_feature).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
ds = ds.batch(batch_size).repeat(num_epochs)
println(ds)