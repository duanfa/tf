import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    with tf.name_scope('layer'):
        layer_name = "layer%s"%n_layer
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name="w")
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'/biases',biases)
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

x_datas = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.1,x_datas.shape)
y_datas =  np.square(x_datas)-0.5 + noise

with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')

l1=add_layer(xs,1,10,1,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,2)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]),name='loss_name')
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
merge = tf.summary.merge_all()
writer = tf.summary.FileWriter("/Users/duanfa/Documents/Nutstore/tf/morvan/tensorflowboard/",sess.graph)
sess.run(init)


for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_datas,ys:y_datas})
    if i%50==0:
        result = sess.run(merge,feed_dict={xs:x_datas,ys:y_datas})
        writer.add_summary(result,i)
            
            
        