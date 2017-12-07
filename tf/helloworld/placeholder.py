import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32, [None, 2])  
y = x
  
with tf.Session() as sess:  
    rand_array = np.random.rand(6, 2)  
    print("array:%s"%rand_array)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed. 