import tensorflow as tf
x =  tf.ones(shape=(2,1))
print (x)

x = tf.zeros(shape=(2,1))
print (x)

x = tf.random.normal(shape= (3,1), mean=0, stddev=1.)
print(x)

x = tf.random.uniform(shape= (3,1), minval=0., maxval=1.)
print(x)

import numpy as np

x = np.ones(shape=(2,2))
x[0,0] = 0.
print(x)

#x = tf.ones(shape=(2,2))
#x[0,0] = 0.

v = tf.Variable(initial_value=tf.random.normal(shape=(3,1)))
print(v)

v.assign(tf.ones((3,1)))
print(v)

v[0,0].assign(3.)
print(v)

v.assign_add(tf.ones((3,1)))
print(v)

a = tf.ones((2,2))
b = tf.square(a)
c = tf.sqrt(a)
d = b+c
e = tf.matmul(a,b)
e *= d

print(b,"/n", c, '/n', d,'n',e)

input_var = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    result = tf.square(input_var)
gradient = tape.gradient(result, input_var)

input_const = tf.constant(3.)
with tf.GradientTape() as tape:
    tape.watch(input_const)
    result = tf.square(input_const)
gradient = tape.gradient(result, input_const)

time = tf.Variable(0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time **2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0,3],
    cov = [[1,0.5],[0.5,1]],
    size = num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3,0],
    cov =[[1,0.5],[0.5,1]],
    size = num_samples_per_class)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

targets = np.vstack((np.zeros((num_samples_per_class,1), dtype='float32'),
                    np.ones((num_samples_per_class,1), dtype='float32')))

import matplotlib.pyplot as plt

plt.scatter(inputs[:,0], inputs[:,1], c=targets[:,0])
plt.show()

input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim,output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

def model(inputs):
    return tf.matmul(inputs, W) +b

def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)

learning_rate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W,b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

for step in range(40):
    loss = training_step(inputs, targets)
    print(f'{step}번째 스텝의 손실:  {loss:.4f}')
