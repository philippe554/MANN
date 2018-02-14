# Memory Augmented Neural Network

This package allows you to make a custom Memory Augmented Neural Network (MANN) by combining different architectures proposed by different papers.

## Getting Started

Packages needed:

* Python 3 
* Numpy
* Tensorflow

### Setup

The model is setup ready to run, no need to change anything. But if you want, next paragraph explains what can be changed.

First define a MANN in the main.py file as follows (Multiple controllers are put in series, multiple heads are put in parallel):

```
cell = mann.MANNUnit("L1MANN")
cell.addMemory(mann.BasicMemory("M1", 20, 12))
cell.addController(mann.FFCell("Controller1", 32))
cell.addHead(mann.DNCHead("Head1", 1))
```

Next create a Generator, this is a class that generates training data and contains the corrosponding settings for the network (Input/output size, entropy, ...)

```
generator = mann.Copy(10,8)
```

Next define your hyper parameters, default ones are fine in most cases

```
TrainSetSize = 10000
TestSetSize = 1000
BatchSize = 100
TrainSteps = 100
```

Finnaly define your optimizer

```
optimizer = tf.train.RMSPropOptimizer(0.001)
```

## Code Structure

UML Diagram of the code

![Alt text](NTM/UML/classes.jpg?raw=true "Title")

## Papers used

### Neural Turing machine

Add a read and write head to the MANN:

```
cell.addHead(mann.NTMHead("Head1"))
```

This head is based on the paper:

Alex Graves et Al. Neural Turing Machine. 2014. https://arxiv.org/abs/1410.5401

### Differentiable Neural Computer

Add a read and write head to the MANN (Where the second parameter defines the amount of reading heads):

```
cell.addHead(mann.DNCHead("Head1", 1))
```

This head is based on the paper:

Alex Graves et Al. Hybrid computing using a neural network with dynamic external memory. 2016. https://www.nature.com/articles/nature20101

### Least Recently Used Acces

*This head is still in production*

Add a read and write head to the MANN:

```
cell.addHead(mann.NTMHead("Head1"))
```

This head is based on the paper:

Adam Santoro et Al. One-shot Learning with Memory-Augmented Neural Networks. 2016. https://arxiv.org/abs/1605.06065

