#coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import ProgressBar as pb


hop = 71
timestep_size = 71      # Hours of looking ahead
output_parameters = 3   # Number of predicting parameters
num_stations = 3        # Number of monitoring stations

training_epochs = 200   # Number of training Epochs
total_batch = 12288     # Number of training instance


data_dir = "./dev_data/"
def process(x):
    if x == '?':
        return 0.0
    else:
        return float(x)

dataset = []
split = 200
leap = 6
length = 16

lr = 0.0001
hidden_size = 256
layer_num = 3


# from UNIX time 1395691200
# to UNIX time 1448564400
# Read from the file of the training set
data = []
target_set = []

# defines how many hours is used to predict

# --------------------------------------------
#             Data Preparation
# --------------------------------------------

print("Processing target set")
start = 1395691200
end = 1448564400
cur_start = start
cur_end = start+hop*3600
count = 0

bar = pb.ProgressBar(total=(end-start)/3600)
while(cur_end < end-(120+288)*3600):
    bar.move()
    count += 1
    if(count % 100 == 0):
        bar.log('Preparing : ' + str(cur_end) + ' till 1448564400')
    buff = []
    for i in range(hop):
        hour = []
        f1 = open(data_dir+(str)(cur_start+i*3600), 'rb')
        for line in f1.readlines():
            ls = line.split('#')
            hour = hour+(map(float, ls[4:16]))
        f1.close()
        buff.append(hour)
    data.append(buff)
    f1 = open(data_dir+(str)(cur_start+120*3600), 'rb')
    for line in f1.readlines():
        ls = line.split("#")
        target_set.append(map(float, ls[7:10]))
        break
    cur_start = cur_start+3600
    cur_end = cur_end+3600
print(len(target_set))
np_data = np.asarray(data)
np_target = np.asarray(target_set)
print("Target shape :", np_target.shape)
print("Data shape : :", np_data.shape)


X = np_data
y = np_target

training_set = np.array(X[1920:])
training_target = np.array(y[1920:])
val_set = np.array(X[:1920])
val_target = np.array(y[:1920])

sess = tf.InteractiveSession()
batch_size = tf.placeholder(tf.int32)
# None corresponds to the batch size, can be of any size
_X = tf.placeholder(tf.float32,[None, 71, 36])      # Add here
y = tf.placeholder(tf.float32, [None, 3])       # Add here
# tf.reshape(_X, [-1])
keep_prob = tf.placeholder(tf.float32)

# --------------------------------------------
#        Construct MLP computation graph
#  --------------------------------------------

in_units = 2556
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([batch_size, in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([batch_size, h1_units, 3]))
b2 = tf.Variable(tf.zeros([3]))

h1 = tf.nn.relu(tf.matmul(_X, W1) + b1)
h1_drop = tf.nn.dropout(h1, keep_prob)
y_ = tf.nn.softmax(tf.matmul(h1_drop, W2) + b2)

# --------------------------------------------
#           Construct Training Algo
# --------------------------------------------

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast((correct_prediction, tf.float32)))

optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

# --------------------------------------------
#               Start Training
# --------------------------------------------

avg_cost = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for i in range(total_batch):
            _, c = sess.run([optimizer, cross_entropy],
                            feed_dict={_X: training_set,
                                       y:training_target,
                                       batch_size: 384,
                                       keep_prob: 1})
            avg_cost += c / total_batch

        if (epoch % 5):
            acc = accuracy.eval({_X: data[1152:1536], y: target_set[1152:1536], batch_size: 384, keep_prob: 0.5})
            print("Epoch: " + str(epoch) + "Acc: " + str(acc))