import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)

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
        Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs



with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32,[None,64],name='x_input')
    ys = tf.placeholder(tf.float32,[None,10],name='y_input')
    keep_prob = tf.placeholder(tf.float32)

l1=add_layer(xs,64,50,'L1',activation_function=tf.nn.tanh)
prediction = add_layer(l1,50,10,'L2',activation_function=tf.nn.softmax)

with tf.name_scope('loss'):
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
    tf.summary.scalar('loss',cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
merge = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("/Users/duanfa/Documents/Nutstore/tf/morvan/tensorflowboard/drop/train",sess.graph)
test_writer = tf.summary.FileWriter("/Users/duanfa/Documents/Nutstore/tf/morvan/tensorflowboard/drop/test",sess.graph)
sess.run(init)


for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.6})
    if i%50==0:
        train_result = sess.run(merge,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        test_result = sess.run(merge,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)
            
            
        