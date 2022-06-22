# Quick Start

This tutorial introduces how to quickly get started with PaddlePaddle through a classic example.

## 1. Installing the Paddle

If you installed paddle before, you can skip this step. You can select your OS and Package to get the corresponding installation commands, which can be viewed in [Quick Installation](https://www.paddlepaddle.org.cn/install/quick).

## 2. import paddle

You can import paddle in the Python program after installing paddle successful.


```python
import paddle
print(paddle.__version__)
```

    2.2.1


## 3. Quick Start: Handwritten Digit Recognition

In brief, deep learning tasks are generally divided into following steps: 1. dataset preparation and loading; 2. model construction; 3. training; 4. evaluation. 


In the following, you can use Paddle to implement the above steps.

### 3.1 Loading the Dataset

Paddle provides some public datasets such as MNIST and FashionMNIST. In this tutorial, you can load two datasets, one for training and the other for validation.


```python
import paddle.vision.transforms as T
transform = T.Normalize(mean=[127.5], std=[127.5], data_format='CHW')

# download dataset
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
val_dataset =  paddle.vision.datasets.MNIST(mode='test', transform=transform)
```

### 3.2 Build the Model

The model is built by `Sequential`. Note that the Tensor needs to be flattened from [1, 28, 28] to [1, 784].


```python
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)
```

### 3.3 Training

Before training, you need set the loss function and optimization function through `model.prepare`. After that you can use `model.fit` to start training.


```python
model = paddle.Model(mnist)

# set loss function and optimization function
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
                paddle.nn.CrossEntropyLoss(),
                paddle.metric.Accuracy())

# traing model
model.fit(train_dataset,
            epochs=5,
            batch_size=64,
            verbose=1)
```

    The loss value printed in the log is the current step, and the metric is the average value of previous steps.
    Epoch 1/5
    step 938/938 [==============================] - loss: 0.1683 - acc: 0.9496 - 8ms/step        
    Epoch 2/5
    step 938/938 [==============================] - loss: 0.1299 - acc: 0.9605 - 8ms/step         
    Epoch 3/5
    step 938/938 [==============================] - loss: 0.0866 - acc: 0.9646 - 9ms/step        
    Epoch 4/5
    step 938/938 [==============================] - loss: 0.0454 - acc: 0.9681 - 8ms/step         
    Epoch 5/5
    step 938/938 [==============================] - loss: 0.2179 - acc: 0.9707 - 7ms/step         


### 3.4 Evaluation

You can use validation dataset to evaluate the accuracy of the model.


```python
model.evaluate(val_dataset, verbose=0)
```




    {'loss': [9.5367386e-07], 'acc': 0.9734}



The accuracy of model is around 97.5%, you can improve the accuracy of the model by adjusting the parameters after get familiar with Paddle. The [Paddle website](https://www.paddlepaddle.org.cn/) provides a wealth of tutorials and guides.
