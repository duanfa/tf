import numpy as np
import tensorflow as tf


in_size=1
out_size=10

Weights = tf.Variable(tf.random_normal([in_size,out_size]))

inputs = np.linspace(-1,1,300)[:,np.newaxis]

Wx_plus_b = tf.matmul(inputs,Weights)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(Wx_plus_b.shape)
sess.run(Wx_plus_b)