import datetime
import os

class epochLogger:
    def __init__(self, name):
        name = name.replace("<TimeStamp>", datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))

        logPath = os.path.join(os.getcwd(), os.pardir, "logging")

        if not os.path.exists(logPath):
            os.makedirs(logPath)

        file = os.path.join(logPath, name)

        self.file = open(os.path.abspath(file), 'w')

        self.file.write("epoch,acc,trainLoss,testLoss\n")

    def log(self, epoch, acc, trainLoss, testLoss):
        self.file.write(str(epoch) + ',' + str(acc) + ',' + str(trainLoss) + ',' + str(testLoss) + '\n')
        self.file.flush()