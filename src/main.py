import tensorflow as tf
import time
import mann
import helper

preCell = mann.LSTMCell("preLSTM", 20)

# Define the MANN
cell = mann.MANNUnit("L1MANN")
cell.addMemory(mann.BasicMemory("M1", 20, 14))
#cell.addController(mann.GRUCell("Controller1", 32))
cell.addController(mann.LSTMCell("LSTM", 40))
#cell.addController(mann.FFCell("FF1", 40))
#cell.addController(mann.FFCell("FF2", 40))
cell.addHead(mann.DNCHead("Head1", 2))
#cell.addHead(mann.NTMHead("Head1"))

#cell = mann.LSTMCell("LSTM1", 40)

# Define the test data
# generator = mann.MinPath(7, 10, 4, 8)
# generator = mann.Copy(10,8)
generator = mann.VertexCover(7, 10, 4, 40)

# Define constants
TrainSetSize = 100000
TestSetSize = 10000
BatchSize = 100
TrainSteps = 100
TestBatchSize = 100
SaveInterval = 50

# Define optimizer
#optimizer = tf.train.RMSPropOptimizer(0.005, decay=0.99)
optimizer = tf.train.AdamOptimizer()
#ptimizer = tf.train.AdadeltaOptimizer(0.01)

loadFromFile = None #"2018-05-04 15-14-33 Epoch-50 Loss-0.ckpt"

logger = mann.epochLogger("<TimeStamp>")

#### End of configuration ####

# Build the network
x = generator.getInput()
h = x
#h = preCell.build(h, generator.inputLength*[1], generator.inputSize)
h = cell.build(h, generator.outputMask, generator.outputSize)
y = h
_y = generator.getLabel()

# Build optimizer
trainStep, p, accuracy, loss = generator.postBuild(_y, y, optimizer)

# Visualize parameters
helper.printStats(tf.trainable_variables())

# Generate the data
trainData = generator.makeAndSaveDataset(TrainSetSize, "train")
testData = generator.makeAndSaveDataset(TestSetSize, "test")
print("Data ready")

# Print class distribution
print("Class distribution:", trainData.C)

# Train network
with tf.Session() as sess:
    if loadFromFile is not None:
        generator.restore(sess, loadFromFile)
    else:
        sess.run(tf.global_variables_initializer())

    X, Y = testData.getBatch(TestBatchSize)
    acc, testLoss = sess.run([accuracy, loss], feed_dict={x: X, _y: Y})
    out = "Start network: acc: " + helper.strfixedFloat(acc, 4, 2)
    out += ", Loss: " + helper.strfixedFloat(testLoss, 7, 4)
    print(out)

    print("Training: ")
    avgAcc = 0
    avgTrainLoss = 0
    avgTrainLossHist = []
    for i in range(100000):
        # Train 1 epoch
        start_time = time.time()
        trainLoss = 0
        for j in range(TrainSteps):
            X, Y = trainData.getBatch(BatchSize)
            _, l = sess.run([trainStep, loss], feed_dict={x: X, _y: Y})
            trainLoss += l
        trainTime = time.time() - start_time
        trainLoss = trainLoss / TrainSteps
        avgTrainLoss += trainLoss

        # Get accuracy
        X, Y = testData.getBatch(TestBatchSize)
        start_time = time.time()
        acc, testLoss, r = sess.run([accuracy, loss, p], feed_dict={x: X, _y: Y})
        testTime = time.time() - start_time
        avgAcc += acc

        out = helper.strfixed("#" + str(i), 5) + " acc: " + helper.strfixedFloat(acc, 4, 2)
        out += " ║ Loss: " + helper.strfixedFloat(trainLoss, 7, 4) + ", " + helper.strfixedFloat(testLoss, 7, 4)
        out += " ║ Time: " + helper.strfixedFloat(trainTime, 5, 2) + "s, " + helper.strfixedFloat(testTime, 5, 2) + "s "
        out += " ║ " + generator.process(X, Y, r)
        print(out)

        logger.log(i, acc, trainLoss, testLoss)

        if i % SaveInterval == 0 and i > 0:
            out = "Avg Acc: " + helper.strfixedFloat(avgAcc / SaveInterval, 5, 3)
            out += ", Avg train loss: " + helper.strfixedFloat(avgTrainLoss / SaveInterval, 8, 5)
            out += ", Loss History:"

            avgTrainLossHist.append(avgTrainLoss / SaveInterval)

            for i in range(10):
                if i < len(avgTrainLossHist):
                    out += " " + helper.strfixedFloat(avgTrainLossHist[-i-1], 6, 4)

            print(out)

            avgAcc = 0
            avgTrainLoss = 0

            generator.save(sess, i, int(trainLoss))


