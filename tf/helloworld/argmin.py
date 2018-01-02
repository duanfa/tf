import tensorflow as tf
import numpy as np

# array=range(24);
# arrayv=tf.Variable(array)
# shuffled=tf.random_shuffle(arrayv)
# matrix=tf.reshape(shuffled,[2,3,4])
m=[[[ 3 ,20, 14 ,23],
  [ 2 ,19 ,17, 22],
  [18, 21, 15 , 4]],

 [[ 8, 10 ,11,  9],
  [16,  1,  0, 13],
  [ 6 ,12,  7,  5]]]
matrix=tf.Variable(m)
sess=tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
matrix_value=sess.run(matrix)
# print(matrix_value)
a=tf.argmin(matrix_value,0)
b=tf.argmin(matrix_value,1)
c=tf.argmin(matrix_value,2)

av,bv,cv=sess.run([a,b,c])
print(av)
print("")
print(bv)
print("")
print(cv)