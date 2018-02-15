# Memory Augmented Neural Network

This package allows you to make a custom Memory Augmented Neural Network (MANN) by combining different architectures proposed by different papers. It is fully modular, and can be added to any other RNN in Tensorflow.

## Features

* 3 types of contollers
* 2 types of heads
* modular
* compatible with batch training
* generate random toy data to train a model


## Getting Started

Packages needed:

* Python 3 
* Numpy
* Tensorflow

Import this package:

```
import mann
```

### Setup

The model is setup ready to run, no need to change anything. Run the main.py file to start training. The next paragraph explains what can be changed if needed.

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

### Use MANN as a layer in a bigger network

First define a MANN as describes above, next make a layer:

```
y = cell.build(x, mask, outputSize)
```

where

* x: the input of the layer with size (BatchSize, len(mask), ?)
* mask: determains which time steps are used to create the output (See example below)
* outputSize: the size of the last dimention of the output
* y: the output of the layer with size (BatchSize, amount of ones in mask, outputSize)

Note: there has not yet been a non linearity applied to y

Example on the mask parameter:

If mask is

```
mask = [0,0,0,1,1,1]
```

Then your input tensor has 6 time steps, and your output tensor has 3 timesteps. The last 3 outputs of the RNN/MANN are used to make the y

## Code Structure

UML Diagram of the code

![Alt text](UML/classes.jpg?raw=true "UML")

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

*This head is still in development*

Add a read and write head to the MANN:

```
cell.addHead(mann.NTMHead("Head1"))
```

This head is based on the paper:

Adam Santoro et Al. One-shot Learning with Memory-Augmented Neural Networks. 2016. https://arxiv.org/abs/1605.06065

