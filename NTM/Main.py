import tensorflow as tf
import time
import mann
import helper

#Define the MANN
cell = mann.MANNUnit("L1MANN")
cell.addMemory(mann.BasicMemory("M1", 20, 12))
#cell.addController(mann.GRUCell("Controller1", 32))
cell.addController(mann.FFCell("Controller1", 32))
cell.addHead(mann.DNCHead("Head1", 1))

#Define the test data
#generator = mann.MinPath(15, 20, 5, 4)
generator = mann.Copy(10,8)

#Define constants
TrainSetSize = 10000
TestSetSize = 1000
BatchSize = 100
TrainSteps = 100

#Define optimizer
optimizer = tf.train.RMSPropOptimizer(0.001)

#### End of configuration ####

#Build the network
x = tf.placeholder(tf.float32, shape=(None, generator.inputLength, generator.inputSize))
_y = tf.placeholder(tf.float32, shape=(None, generator.outputLength, generator.outputSize))
y = cell.build(x, generator.outputMask, generator.outputSize)

#Build optimizer
trainStep, p, accuracy, loss = generator.postBuild(_y, y, optimizer)

#Visualize parameters
helper.printStats(tf.trainable_variables())

#Generate the data
print("Start generating data")
trainData = generator.makeDataset(TrainSetSize)
testData = generator.makeDataset(TestSetSize)
print("Finished generating data")

#Print class distribution
print(trainData.C)

#Train network
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #Get accuracy before training
    X,Y = testData.getBatch(BatchSize)
    acc, testLoss = sess.run([accuracy, loss], feed_dict={x: X, _y: Y})
    print("Start:" + "\tacc: " + str(acc) + "\tLoss: " + str(testLoss))

    for i in range(100000):
        #Train 1 epoch
        trainLoss=0
        start_time = time.time()
        for j in range(TrainSteps):
            X,Y = trainData.getBatch(BatchSize)
            _, l = sess.run([trainStep, loss], feed_dict={x: X, _y: Y})
            trainLoss+=l
        duration = time.time() - start_time
        trainLoss = trainLoss/BatchSize

        #Get accuracy
        X,Y = testData.getBatch(BatchSize)
        acc, testLoss = sess.run([accuracy, loss], feed_dict={x: X, _y: Y})

        #Print data
        print("#" + str(i+1) + "\tacc: " + str(acc) + "\tLoss: " + str(trainLoss) + "-" + str(testLoss) + "\tTime: " + "{0:.4f}".format(duration) + "s")
        
