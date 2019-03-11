#referenced by https://github.com/aymericdamien/TensorFlow-Examples/
#using tensorflow to build the neural network for chess training
from __future__ import division, print_function, absolute_import
import sys
import tensorflow as tf
import numpy as np
import sklearn.preprocessing
from sklearn.utils import shuffle

###
#TODO
#1.add a better model using tensorflow
#2.write a pyTorch or Keras version
###

#load data from numpy
with np.load("trainData/trainData1m.npz") as data:
  features = data["X"]
  labels = data["y"]

(features, labels) = shuffle(features, labels, random_state=0)

#one hot encode labels
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(max(labels)+1))
labels = label_binarizer.transform(labels)

trainSize = len(features)
trainPer = 0.9
assert features.shape[0] == labels.shape[0]

Relu = tf.nn.relu
Tanh = tf.nn.tanh
BatchNormalization = tf.layers.batch_normalization
Dropout = tf.layers.dropout
Dense = tf.layers.dense

num_input = 320   #5*8*8 = 320
num_output = 3
epochs = 1000
batch_size = 256
learning_rate = 0.01
display_step = 10

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_output])
keep_prob = tf.placeholder(tf.float32)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 8, 8, 5])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], strides=2)
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'], strides=2)
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'])
    conv8 = conv2d(conv7, weights['wc8'], biases['bc8'])
    #conv9 = conv2d(conv8, weights['wc9'], biases['bc9'], strides=2)

    # Max Pooling (down-sampling)
    #maxp1 = maxpool2d(conv2, k=2)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv8, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 5 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([4, 4, 5, 16])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([4, 4, 16, 16])),
    # 5x5 conv, 5 input, 32 outputs
    'wc3': tf.Variable(tf.random_normal([4, 4, 16, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc4': tf.Variable(tf.random_normal([4, 4, 32, 32])),
    # 5x5 conv, 5 input, 32 outputs
    'wc5': tf.Variable(tf.random_normal([4, 4, 32, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc6': tf.Variable(tf.random_normal([4, 4, 32, 64])),
    # 5x5 conv, 5 input, 32 outputs
    'wc7': tf.Variable(tf.random_normal([4, 4, 64, 64])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc8': tf.Variable(tf.random_normal([4, 4, 64, 128])),
    # 5x5 conv, 5 input, 32 outputs
    # fully connected, 2*2*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([2*2*128, 1024])),
    # 1024 inputs, 3 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_output]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([16])),
    'bc3': tf.Variable(tf.random_normal([32])),
    'bc4': tf.Variable(tf.random_normal([32])),
    'bc5': tf.Variable(tf.random_normal([32])),
    'bc6': tf.Variable(tf.random_normal([64])),
    'bc7': tf.Variable(tf.random_normal([64])),
    'bc8': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_output]))
}

# Construct model
with tf.device("/gpu:0"):
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    features = np.reshape(features, (features.shape[0], num_input))
    #labels = np.reshape(labels, (labels.shape[0], 3))
    # Start training
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # Run the initializer
    sess.run(init)
    print("Initialized")

    for step in range(epochs):

        # Generate a minibatch.
        offset = (step * batch_size) % (trainSize - batch_size)
        #print(offset)
        batch_data = features[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size), :]


        #print("batch_data_shape", batch_data.shape)
        #print("batch_labels_shape", batch_labels.shape)

        sess.run(train_op, feed_dict = {X: features,
                                        Y: labels,
                                        keep_prob:0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: features,
                                                                 Y: labels,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for all training data
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: features,
                                      Y: labels,
                                      keep_prob: 1.0}))