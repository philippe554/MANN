import datetime
import os

class epochLogger:
    def __init__(self, name, more=None):
        name = name.replace("<TimeStamp>", datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))

        logPath = os.path.join(os.getcwd(), os.pardir, "logging")

        if not os.path.exists(logPath):
            os.makedirs(logPath)

        file = os.path.join(logPath, name)

        self.file = open(os.path.abspath(file), 'w')

        title = "epoch,acc,trainLoss,testLoss"

        if more != None:
            for e in more:
                title += ',' + e

        self.file.write(title + '\n')

    def log(self, epoch, acc, trainLoss, testLoss, more=None):
        out = str(epoch) + ',' + str(acc) + ',' + str(trainLoss) + ',' + str(testLoss)
        if more != None:
            for e in more:
                out += ',' + str(e)
        self.file.write(out + '\n')
        self.file.flush()