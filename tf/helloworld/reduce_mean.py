import tensorflow as tf

#create tensor
x = tf.Variable([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
#display
init = tf.global_variables_initializer();
with tf.Session() as sess:
    sess.run(init);
    #tf.reduce_mean(input_tensor,  axis=None,  keep_dims=False,  name=None,  reduction_indices=None)
    y = tf.reduce_mean(x);
    y01 = tf.reduce_mean(x, axis=0, keep_dims=False);
    y02 = tf.reduce_mean(x, axis=0, keep_dims=True);
    y1 = tf.reduce_mean(x, axis=1);
    
    print("x = ", x.eval());
    
    #all mean
    print("tf.reduce_mean(x) = ", y.eval());
    
    #x axis mean
    print("tf.reduce_mean(x, axis=0, keep_dims=False) = ", y01.eval());
    print("tf.reduce_mean(x, axis=0, keep_dims=True) = ", y02.eval())
    
    #y axis mean
    print("tf.reduce_mean(x, axis=1) = ", y1.eval());