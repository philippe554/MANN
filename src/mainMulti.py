import tensorflow as tf
import time
import mann
import helper

# Define the MANN
cell = mann.MANNUnit("L1MANN")
cell.addMemory(mann.BasicMemory("M1", 20, 14))
# cell.addController(mann.GRUCell("Controller1", 32))
cell.addController(mann.LSTMCell("LSTM", 40))
cell.addHead(mann.DNCHead("Head1", 2))

# Define the test data
# generator = mann.MinPath(7, 10, 4, 8)
# generator = mann.Copy(10,8)
generator = mann.VertexCover(7, 10, 4)

# Define constants
TrainSetSize = 10000
TestSetSize = 1000
BatchSize = 100
TrainSteps = 100

# Define optimizer
optimizer = tf.train.RMSPropOptimizer(0.001)

loadFromFile = None

#### End of configuration ####

# Build the network
x = tf.placeholder(tf.float32, shape=(None, generator.inputLength, generator.inputSize))
_y = tf.placeholder(tf.float32, shape=(None, generator.possibleRightAnswer, generator.outputLength, generator.outputSize))
y = cell.build(x, generator.outputMask, generator.outputSize)

assert helper.check(_y, [generator.possibleRightAnswer, generator.outputLength, generator.outputSize], BatchSize)
assert helper.check(y, [generator.outputLength, generator.outputSize], BatchSize)

yy = tf.expand_dims(y, axis=-2)
assert helper.check(yy, [1, generator.outputLength, generator.outputSize], BatchSize)

sq = tf.square(tf.subtract(yy, _y))
assert helper.check(sq, [generator.possibleRightAnswer, generator.outputLength, generator.outputSize], BatchSize)

distance = tf.sqrt(tf.reduce_sum(tf.reduce_sum(sq, axis=-1), axis=-1))
assert helper.check(distance, [generator.possibleRightAnswer], BatchSize)

indices = tf.argmin(distance, axis=-1)
assert helper.check(indices, [], BatchSize)

num_examples = tf.cast(tf.shape(y)[0], dtype=indices.dtype)
indices = tf.stack([tf.range(num_examples), indices], axis=-1)
Y = tf.gather_nd(_y, indices)
assert helper.check(Y, [generator.outputLength, generator.outputSize], BatchSize)

Y = tf.stop_gradient(Y)

# Build optimizer
trainStep, p, accuracy, loss = generator.postBuild(Y, y, optimizer)

# Visualize parameters
helper.printStats(tf.trainable_variables())

# Generate the data
trainData = generator.makeAndSaveDataset(TrainSetSize, "train")
#testData = generator.makeAndSaveDataset(TestSetSize, "test")
print("Data ready")

# Print class distribution
print(trainData.C)

# Train network
with tf.Session() as sess:
    if loadFromFile is not None:
        generator.restore(sess, loadFromFile)
    else:
        sess.run(tf.global_variables_initializer())

    # Get accuracy before training
    X, Y = trainData.getBatch(BatchSize)
    acc, testLoss = sess.run([accuracy, loss], feed_dict={x: X, _y: Y})
    print("Start:" + "\tacc: " + str(acc) + "\tLoss: " + str(testLoss))

    for i in range(100000):
        # Train 1 epoch
        trainLoss = 0
        start_time = time.time()
        for j in range(TrainSteps):
            X, Y = trainData.getBatch(BatchSize)
            _, l = sess.run([trainStep, loss], feed_dict={x: X, _y: Y})
            trainLoss += l
        duration = time.time() - start_time
        trainLoss = trainLoss / BatchSize

        # Get accuracy
        X, Y = trainData.getBatch(BatchSize)
        acc, testLoss = sess.run([accuracy, loss], feed_dict={x: X, _y: Y})

        # Print data
        print("#" + str(i + 1) + "\tacc: " + str(acc) + "\tLoss: " + str(trainLoss) + "-" + str(
            testLoss) + "\tTime: " + "{0:.4f}".format(duration) + "s")

        if i % 50 == 0 and i > 0:
            generator.save(sess, i, int(trainLoss))


