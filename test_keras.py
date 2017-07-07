import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import layers


K.set_session(tf.get_default_session())

img = tf.placeholder(tf.float32, shape=(None, 784))
x = layers.Dense(128, activation='relu')(img)
print x.trainable_weights
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(10, activation='softmax')(x)



labels = tf.placeholder(tf.int32)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds, labels=labels))

mnist_data = input_data.read_data_sets("MNIST_data", one_hot=False)
optimize = tf.train.RMSPropOptimizer(0.001).minimize(loss)

acc = tf.reduce_mean(tf.cast(tf.not_equal(tf.cast(tf.argmax(preds, 1), tf.int32), labels), tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session().as_default() as sess:
    sess.run(init_op)
    for i in range(10):
        for j in range(100):
            batch = mnist_data.train.next_batch(100)
            sess.run(optimize, feed_dict={img: batch[0], labels: batch[1], K.learning_phase(): 1})
        print sess.run(acc, feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels, K.learning_phase(): 0})

