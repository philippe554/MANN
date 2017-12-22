import tensorflow as tf

def map(name, input, outputSize, r=true):
    with tf.variable_scope(name, reuse=r):
        inputSize = input.get_shape()[0]
        m = tf.get_variable(name+"M", tf.random_normal([int(inputSize),int(outputSize)]))
        i1 = tf.reshape(input, [1,-1])
        i2 = tf.matmul(i1, m)
        b = tf.get_variable(name+"B", tf.random_normal(i2.get_shape()))
        i3 = i2 + b
        return tf.reshape(i3, [-1])

def makeStartState(shape):
    with tf.variable_scope("init"):
        product = 1
        for i in shape:
            product *= i

        C = tf.constant([[1]])
        O = tf.tanh(map("map", C, product, false))
        return tf.reshape(O, shape)