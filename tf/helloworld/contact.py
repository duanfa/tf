import tensorflow as tf

t1 = [[1, 2, 3], [4, 5, 6]]  
t2 = [[7, 8, 9], [10, 11, 12]]
# tf.concat(1, [t1, t2])
print(tf.concat(t1,0))
print(tf.concat([t1,t2],0))