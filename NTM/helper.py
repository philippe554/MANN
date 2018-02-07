import tensorflow as tf
import numpy as np
import random

def mapOld(name, input, outputSize, r=tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=r):
        inputSize = int(input.get_shape()[0])
        m = tf.get_variable("M", initializer=tf.random_normal([inputSize,int(outputSize)]))
        i1 = tf.reshape(input, [1,-1])
        i2 = tf.matmul(i1, m)
        b = tf.get_variable("B", initializer=tf.random_normal(i2.get_shape()))
        i3 = i2 + b
        return tf.reshape(i3, [-1])

def map(name, input, outputSize, r=tf.AUTO_REUSE):
    if(len(input.get_shape())==1):
        with tf.variable_scope(name, reuse=r):
            inputSize = input.get_shape()[0]
            input = tf.expand_dims(input, 0)
            m = tf.get_variable("M", initializer=tf.random_normal([inputSize,int(outputSize)]))
            b = tf.get_variable("B", initializer=tf.random_normal([int(outputSize)]))
            return tf.squeeze(tf.matmul(input, m) + b, [0])
    else:
        with tf.variable_scope(name, reuse=r):
            inputSize = input.get_shape()[1].value
            m = tf.get_variable("M", initializer=tf.random_normal([inputSize,int(outputSize)]))
            b = tf.get_variable("B", initializer=tf.random_normal([int(outputSize)]))
            return tf.matmul(input, m) + b

def makeStartState(name, shape):
    with tf.variable_scope("init"):
        product = 1
        for i in shape:
            product = product * i

        C = tf.constant([[1]], dtype=tf.float32)
        O = tf.tanh(map(name, C, product, False))
        return tf.reshape(O, shape)

def makeStartStateBatch(name, batchSize, shape):
    with tf.variable_scope("init"):
        product = 1
        for i in shape:
            product = product * i

        C = tf.constant([[1]], dtype=tf.float32)
        O = tf.tanh(map(name, C, product, False))
        return tf.expand_dims(tf.ones(shape, tf.float32), 0) * O

def printStats(variables):
    print("Trainable variables:")
    total_parameters = 0
    for variable in variables:
        shape = variable.get_shape()
        print(str(variable.name)+": "+str(shape))
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Number of parameters: " + str(total_parameters))

def getTrainableConstant(name, size, batches=None):
    state = tf.get_variable(name, initializer=tf.random_normal([int(size)]))

    if batches is not None:
        state = tf.reshape(tf.tile(state, [batches]), [batches, int(size)])

    return state

def check(t, shape, batchSize):
    if batchSize is not None:
        shape = [None] + shape

    if(len(t.get_shape())!=len(shape)):
        return False
    for i,v in enumerate(shape):
        if(v is None and t.get_shape().as_list()[i] is not None):
            return False
        if(v is not None and t.get_shape().as_list()[i] is None):
            return False
        if(v is not None and t.get_shape().as_list()[i] is not None):
            if(t.get_shape().as_list()[i] != v):
                return False
    return True



##### OLD CODE THAT I NEED TO STORE SOMEWHERE #####

#tf.summary.scalar('loss', loss)
#tf.summary.scalar('accuracy', accuracy)

#for grad, var in grads_and_vars:
#    if grad is not None:
#        tf.summary.histogram("grad/"+var.name, grad)

 #writer.add_summary(summary, i)

#merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter("C:/temp/tf_log/", sess.graph)
    # python -m tensorboard.main --logdir="C:/temp/tf_log/"
    # localhost:6006

    #plt.ion()



#print("#" + str(i+1) + "\tacc: " + "{0:.4f}".format(acc) + "\tLoss: " + str(int(trainLoss)) + "-" + str(int(testLoss)) + "\tTime: " + "{0:.4f}".format(duration) + "s")

        #if(i%1==0):
        #    X,Y = helper.getNewxyBatch(length, bitDepth, 1)
        #    acc, w = sess.run([accuracy, W], feed_dict={x: X, _y: Y})
        #    plt.imshow(w, vmin=0, vmax=1);
        #    plt.show()
        #    plt.pause(0.05)