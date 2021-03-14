import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout  batch size None是未定义的
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        #这个dropout直接往里面传值，不用指定batch  size
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        #生成tensor,常量矩阵
        l2_loss = tf.constant(0.0)

        # Embedding layer  指定设备 和编号   命令域
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                #指定初始化的范围   vocab_size是10000多不重复的  embedding_size是128的维度
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            #拿w和input.x
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #转化为4维的，原本是三维的，tf处理的是4维的，新维度是-1
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        #创建每个过滤器的大小卷积+ maxpool层
        pooled_outputs = []
        #3,4,5,
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                #filter_size选几个单词h，embedding_size每个占了多长w   7*5*1  输入1维，输出128维 128个特针图
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                #高斯初始化
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #初始化为常量0.1
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    #不做padding
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    #(len-fiter+padding)/strides =len-filter
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        #合并所有汇集的特性
        num_filters_total = num_filters * len(filter_sizes)
        #self.h_pool = tf.concat(pooled_outputs, 3)
        #self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool=tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # 正则化L2计算损失
            #利用 L2 范数来计算张量的误差值，但是没有开方并且只取 L2 范数的值的一半
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # wx+b
            #logits = tf.matmul(self.h_drop, weights) + biases
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")