#encoding=utf8
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
#读取整个输入样本的训练集，测试集及标签文件
mnist = input_data.read_data_sets('data\\', one_hot=True)
#28*28*1
import tensorflow as tf
#import tensorflow.contrib.eager as tfe
#pip  install tf-nightly
import os
import numpy as np
lr=0.001
training_iters=10000
batch_size=64
display_step=1

input_dim=784
mnist_class=10
dropout=0.5


x=tf.placeholder(tf.float32,[None,input_dim])
y=tf.placeholder(tf.float32,[None,mnist_class])
drop_prob=tf.placeholder(tf.float32)

def conv2d(name,input_data,input_filter,bias):
    x=tf.nn.conv2d(input_data,input_filter, strides=[1,1,1,1], padding='SAME', use_cudnn_on_gpu=False, 
                  data_format="NHWC", name=None)
    x=tf.nn.bias_add(x, bias, data_format=None, name=None)
    return tf.nn.relu(x,name=name)
def max_pooling(name,input_data,k):
    #ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    #strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
    return tf.nn.max_pool(input_data, ksize=[1,k,k,1],strides=[1,k,k,1], padding='SAME',name=name)

def norm(name,input_data,lsize=4):
    return tf.nn.lrn(input_data, depth_radius=lsize, bias=1, alpha=1, beta=0.5, 
                    name=name)
weights={
    # 卷积核filter大小11*11 输入层为1个feature maps，输出层有64 feature maps
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 48])),
    # 卷积核filter大小5*5 输入层为192个feature maps，输出层有384 feature maps
    'wc2': tf.Variable(tf.random_normal([5, 5, 48, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 192])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 192, 192])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 192, 128])),
    'wd1': tf.Variable(tf.random_normal([4*4*128, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'out': tf.Variable(tf.random_normal([4096, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([48])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([192])),
    'bc4': tf.Variable(tf.random_normal([192])),
    'bc5': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([mnist_class]))
}

def alexnet(_input_img,_weights,_biases,_dropout):
    _input_img=tf.reshape(_input_img,shape=[-1,28,28,1])
    #第一层
    conv1=conv2d("conv1_layer", _input_img, _weights['wc1'], _biases['bc1'])
    pooling1=max_pooling("pooling1", conv1, k=2)
    norm1=norm("norm1", pooling1,lsize=4)
    #第二层
    with tf.device('/cpu'):
        conv2=conv2d("conv2_layer",norm1,_weights['wc2'],_biases['bc2'])
        pooling2=max_pooling("pooling2",conv2,k=2)
        norm2=norm("norm2",pooling2,lsize=4)
    #第三层
    conv3=conv2d("conv3_layer",norm2,_weights['wc3'],_biases['bc3'])
    norm3=norm("norm3",conv3,lsize=4)
    #第四层
    conv4=conv2d("conv4_layer",norm3,_weights['wc4'],_biases['bc4'])
    norm4=norm("norm4",conv4,lsize=4)
    #第五层
    conv5=conv2d("conv5_layer",norm4,_weights['wc5'],_biases['bc5'])
    pooling5=max_pooling("pooling5",conv5,k=2)
    norm5=norm("norm5", pooling5,lsize=4)    
    #第六层
    dense1=tf.reshape(norm5,[-1,_weights['wd1'].get_shape().as_list()[0]])
    dense1=tf.nn.relu(tf.matmul(dense1,_weights['wd1'])+_biases['bd1'],name='fc1')
    dense1=tf.nn.dropout(dense1, keep_prob=_dropout)
    
    #第七层
    dense2=tf.reshape(dense1,[-1,_weights['wd2'].get_shape().as_list()[0]])
    dense2=tf.nn.relu(tf.matmul(dense2,_weights['wd2'])+_biases['bd2'],name='fc2')
    dense2=tf.nn.dropout(dense2, keep_prob=_dropout)    
    out = tf.matmul(dense2, _weights['out']) + _biases['out']  # X^T*W+b
    return out


# 构建模型
pred = alexnet(x, weights, biases, drop_prob)

# 定义学习率learning_rate=learning_rate*decay_rate^(global_step/decay_steps)
#第一个参数learning_rate即初始学习速率，第二个参数，是用来计算步骤的，每调用一次优化器，即自增1，
#第三个参数decay_steps通常设为一个常数，如数学公式中所示，与第五个参数配合使用效果较好，
#第五个参数staircase如果设置为True，那么指数部分就会采用整除策略，表示每decay_step，学习速率变为原来的decay_rate
#至于第四个参数decay_rate表示的是学习速率的下降倍率。
global_step=tf.constant(0,tf.int64)
decay_rate=tf.constant(0.9,tf.float64)
learn_rate=tf.train.exponential_decay(lr, global_step, 10000, 
                                     decay_rate)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
# 测试网络
#tf.arg_max(pred,1)是按行取最大值的下标,假如下标一样，返回TRUE，否则返回False
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#先将correct_pred中TRUE和false转换为float32类型,1,0
#求correct_pred中的平均值，因为correct_pred中除了0就是1，因此求平均值即为1的所占比例，即正确率
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的共享变量

init = tf.global_variables_initializer()
# 开启一个训练
def train():
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        #迭代次数不超过规定最大次数（training_iters）
        while step * batch_size < training_iters:
            # 每一次从mnist的训练集中取出batch_size个图片数据，进行训练
            # batch_xs为图片数据 ； batch_ys 为标签值
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) 
            # 获取批数据，开始训练
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,drop_prob: dropout})
            #每次运行display_step=20步，计算精度，计算损失值和打印
            if step % display_step == 0:
                # 计算精度
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, drop_prob: 1.})
                # 计算损失值
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, drop_prob: 1.})
                print ("Iter " + str(step*batch_size) + ", Minibatch Loss = " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
            step += 1
        print ("Optimization Finished!")
        #保存模型
        #初始化一个保存方法
        saver=tf.train.Saver()
        #制定保存文件夹名称
        save_path='ckpt'
        #检查文件夹是否存在，假如不存在文件夹，就创建一个文件夹
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        #制定模型保存的路径下的文件名，
        model_name=save_path+os.sep+"alexnet.ckpt"
        #调用保存方法，sess就是模型的参数和对应的值
        saver.save(sess,model_name)  
        
        #可视化图片

        #img_input=mnist.test.images[img_index:img_index+1,:]
        #predict=sess.run(pred,feed_dict={x:img_input,drop_prob:1.0})
        #Nums = [0,1,2,3,4,5,6,7,8,9]
        
        #print ('prediction is:',np.where(np.int16(np.round(predict[img_index,:],0))==1)[0][0])    
        ##可视化卷积核
        
        # 计算测试精度
        print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], drop_prob: 1.}))
if __name__=='__main__':
    #检测ckpt目录是否存在，存在就开始读取模型文件，开始部署模式
    if os.path.exists("ckpt"):
        count=0
        ls=os.listdir("ckpt")
        #检测目录下是否有4个模型文件，
        for a in ls:
            count+=1
        #如果目录下有4个文件，开始调用模型
        if count ==4:
            init = tf.global_variables_initializer()
            restore=tf.train.Saver()
            with tf.Session() as sess:
                #sess.run(init)     
                ckpt = tf.train.get_checkpoint_state("ckpt")  
                if ckpt and ckpt.model_checkpoint_path:
                    #载入模型
                    restore.restore(sess,ckpt.model_checkpoint_path)
                #制定读取测试文件的索引号
                img_index=10 #  
                #把mnist.test.images测试集里面所有样本转化成图片格式，-1代表文件个数，28,28,1代表一个图片文件
                test_imgs=tf.reshape(mnist.test.images,[-1,28,28,1])#读入测试集所有文件，并且改变图像的形状
                #取索引号为img_index=10的一张图片
                a_img=sess.run(test_imgs)[img_index,:,:,0]
                #取出索引号为img_index的一张图片，img_index:img_index+1表示取一张图片，而且是索引号从16到17的
                img_input=mnist.test.images[img_index:img_index+1,:]
                #表示把pred依赖的输入传给pred变量，drop_prob:1.0表示全部保留神经元，不进行dropout操作
                #pred = alexnet(x, weights, biases, drop_prob)
                predict=sess.run(pred,feed_dict={x:img_input,drop_prob:1.0})
                #tf.argmax表示取predict变量最大元素的索引号，1表示取每一行最大的索引值
                result=tf.argmax(predict,1)
                
                a_img=tf.reshape(img_input,[1,28,28])
                #把预测结果打印出来
                print('prediction is:',sess.run(result))                
                import matplotlib.pyplot as plt
                import pylab
                plt.imshow(a_img)      
                pylab.show()

          
                ###get_ipython().magic('matplotlib inline')
                print (sess.run(weights['wc1']).shape)
                f, axarr = plt.subplots(4,figsize=[10,10])
                axarr[0].imshow(sess.run(weights['wc1'])[:,:,0,0])
                axarr[1].imshow(sess.run(weights['wc2'])[:,:,23,12])
                
                axarr[2].imshow(sess.run(weights['wc3'])[:,:,41,44])
                axarr[3].imshow(sess.run(weights['wc4'])[:,:,45,55])  
                pylab.show()
    else:
        #训练
        train()