from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

X = tf.constant([[2],[-3],[6],[5]], dtype = tf.float32)
y_true = tf.constant([[1],[-2],[8],[4]], dtype =tf.float32)

linear_model = tf.layers.Dense(units=1)
y = linear_model(X)



#print(sess.run(y))

loss = tf.losses.mean_squared_error(labels = y_true, predictions =y)
#print(sess.run(loss))

optimizer = tf.train.GradientDescentOptimizer(0.02)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(300):
    _, loss_value =sess.run((train, loss))
    print(loss_value)

print(sess.run(y))
sess.close()

