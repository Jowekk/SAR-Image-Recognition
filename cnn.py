import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from load_data import one_hot as Y
from load_data import Y_all_one as Y_all
from load_data import X, num_classes, X_all

log_path = './log'

batch_size = 64
train_nums = int(X.shape[0] * 0.8) # 9772
test_nums = X.shape[0] - train_nums # 4188
train_it_max = np.floor(train_nums/batch_size)
test_it_max = np.floor(test_nums/batch_size)

x_train = X[0:train_nums,:,:,:]
y_train = Y[0:train_nums,:]

x_test = X[train_nums:,:,:,:]
y_test = Y[train_nums:]

def SAR_net(inputs, num_classes, is_training, scope='SAR_net'):
    with tf.variable_scope(scope, 'SAR_net'):
        net = slim.conv2d(inputs, 64, [3,3], padding='SAME', scope='conv_1')
        net = slim.max_pool2d(net, 2, stride=2, scope='maxpool_1')
        net = slim.conv2d(net, 128, [2,2], padding='SAME', scope='conv_2')
        net = slim.max_pool2d(net, 2, stride=2, scope='maxpool_2')

        net = slim.flatten(net)
        
        net = slim.fully_connected(net, 512, scope='fc_1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
        net = slim.fully_connected(net, num_classes, activation_fn=None, scope='logits')

    return net

x = tf.placeholder(tf.float32, shape=[None, 8, 8, 6], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')
is_training = tf.placeholder(tf.bool, name='is_training')

logits = SAR_net(x, num_classes, is_training)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cross_entropy)

loss_summary = tf.summary.scalar('loss', cross_entropy)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

#train_writer = tf.summary.FileWriter(log_path, sess.graph)

for i in range(350000):
    next_batch = int(i % train_it_max)
    train_images = x_train[next_batch*batch_size:(next_batch+1)*batch_size,:,:,:]
    train_labels = y_train[next_batch*batch_size:(next_batch+1)*batch_size,:]
    _ = sess.run(train_step, feed_dict={x: train_images, y: train_labels, is_training: True})

    if i % 1000 == 0:
        summary, acc = sess.run([accuracy_summary, accuracy], feed_dict={x: train_images, y: train_labels, is_training: False}) #TODO
        #train_writer.add_summary(summary, i)
        print("Step: %5d, Train Accuracy = %5.2f%%" % (i, acc * 100))

val_acc_all = np.zeros((153, 1))
for j in range(153):
    val_images = X_all[j*1024 :(j+1)*1024, :, :, :]
    val_labels = Y_all[j*1024 :(j+1)*1024, :]
    val_acc_all[j] = sess.run(accuracy, feed_dict={x: val_images, y: val_labels, is_training: False}) #TODO
    print val_acc_all[j]
print("----------ALL ATTENTION-----------")
print("Validation Accuracy = %5.2f%%" % (np.mean(val_acc_all) * 100))


#saver.save(sess, log_path)
