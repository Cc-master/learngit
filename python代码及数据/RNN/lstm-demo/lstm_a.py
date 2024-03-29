#encoding=utf8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#start training
def compute_accuracy(v_x, v_y):
    global pred
    #input v_x to nn and get the result with y_pre
    y_pre = sess.run(pred, feed_dict={x:v_x})
    #find how many right
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y,1))
    #calculate average
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #get input content
    result = sess.run(accuracy,feed_dict={x: v_x, y: v_y})
    return result

def Bi_lstm(X):
    lstm_f_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    lstm_b_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    return tf.contrib.rnn.static_bidirectional_rnn(lstm_f_cell, lstm_b_cell, X, dtype=tf.float32)

def RNN(X,weights,biases):
    # hidden layer for input
    X = tf.reshape(X, [-1, n_inputs])#得到 X.shape=[32*28,28]
    X_in = tf.matmul(X, weights['in']) + biases['in']  #weights['in'].shape=[28,128]，X_in为一个[32*28,128]的矩阵

    #reshape data put into bi-lstm cell
    X_in = tf.reshape(X_in, [-1,n_steps, n_hidden_units])#左边的X_in为一个[32，28,128]的矩阵
    #hape_b=sess.run(X_in)
    #print("the shape before is ",shape_b.shape)
    #print(shape_b)
    X_in = tf.transpose(X_in, [1,0,2])#左边的X_in为一个[28，32,128]的矩阵
    #shape_a=sess.run(X_in)
    #print("the shape after is ",shape_a.shape)
    X_in = tf.reshape(X_in, [-1, n_hidden_units])#左边的X_in为一个[28*32,128]的矩阵
    X_in = tf.split(X_in, n_steps)
    outputs, a, b= Bi_lstm(X_in)#此时X_in是28个[32，128]的矩阵
    
    #hidden layer for output as the final results
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results
    

#load mnist data
mnist = input_data.read_data_sets('C:\\lecture\\code\\data\\', one_hot=True)

# parameters init
l_r = 0.001
training_iters = 1000000
batch_size = 32

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

#define placeholder for input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# define w and b
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out': tf.Variable(tf.random_normal([2*n_hidden_units,n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    'out': tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

pred = RNN(x, weights, biases)
soft_out=tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op = tf.train.AdamOptimizer(l_r).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
cast=tf.cast(correct_pred, tf.float32)


# x_image,x_label = mnist.test.next_batch(500)
# x_image = x_image.reshape([500, n_steps, n_inputs])
#init session
sess = tf.Session()
#init all variables
sess.run(tf.global_variables_initializer())
with tf.device("/gpu:0"):
    for i in range(500):
        #get batch to learn easily
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
        t,p,s,c,cp,cas=sess.run([train_op,pred,soft_out,cost,correct_pred,cast],feed_dict={x: batch_x, y: batch_y})
        if i % 50 == 0:
            print(sess.run(accuracy,feed_dict={x: batch_x, y: batch_y,}))
test_data = mnist.test.images.reshape([-1, n_steps, n_inputs])
test_label = mnist.test.labels
#print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
print("Testing Accuracy: ", compute_accuracy(test_data, test_label))