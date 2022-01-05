# Training and Evaluation

After data preprocessing, data loading and building the model, you can train and evaluate your model.

Paddle provides two methods of training and evaluation, one is to use hapi `paddle.Model`, such as `Model.fit()`, `Model.evaluate()`, `Model.predict()`, etc. to complete the training and evaluation of the model; the other is based on the basic API to train.

## 1. Pre-training Code

Data loading and building model need to be completed before training.


```python
import paddle
from paddle.vision.transforms import ToTensor

# dataset
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

# build the model
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(1, -1),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)
```

## 2. Training with paddle.Model

 you need to pass the model into paddle.Model first to train with `paddle.Model`.


```python
model = paddle.Model(mnist)
```

### 2.1 Using model.prepare define functions

Before training, you need define loss funtion, optimization function, metrics through model.prepare.


```python
model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())
```

### 2.2 Using model.fit train model

Then you can train the model for a fixed number of epochs through model.fit . At least 3 key parameters need to be specified: the training dataset, the number of training epochs and the batch size of dataset.


```python
# training the model
model.fit(train_dataset,
          epochs=5,
          batch_size=64,
          verbose=1)
```

    The loss value printed in the log is the current step, and the metric is the average value of previous steps.
    Epoch 1/5
    step 938/938 [==============================] - loss: 0.1416 - acc: 0.9395 - 22ms/step         
    Epoch 2/5
    step 938/938 [==============================] - loss: 0.0366 - acc: 0.9734 - 23ms/step         
    Epoch 3/5
    step 938/938 [==============================] - loss: 0.0134 - acc: 0.9806 - 24ms/step        
    Epoch 4/5
    step 938/938 [==============================] - loss: 0.0028 - acc: 0.9843 - 23ms/step        
    Epoch 5/5
    step 938/938 [==============================] - loss: 0.1386 - acc: 0.9870 - 23ms/step        


### 2.3  Using model.evaluate to evaluate model

The model can be evaluated by calling model.evaluate directly.


```python
eval_result = model.evaluate(test_dataset, verbose=1)
```

    Eval begin...
    step 10000/10000 [==============================] - loss: 1.7047e-05 - acc: 0.9807 - 3ms/step         
    Eval samples: 10000


### 2.4  Using model.predict to predict

You can use model.predict to predict with your data.


```python
test_result = model.predict(test_dataset)
```

    Predict begin...
    step 10000/10000 [==============================] - 2ms/step        
    Predict samples: 10000


## 3. Training and Evaluation with basic API

You can train and evaluate your model with basic API. The following breaks down the hapi to the basic API to show how to complete the training and evaluation of the model.

### 3.1 Training model with basic API

Training model with basic API，corresponding to Model.prepare and Model.fit.


```python
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

mnist.train()

# define epochs
epochs = 5

# define optimizer
optim = paddle.optimizer.Adam(parameters=mnist.parameters())
# define loss
loss_fn = paddle.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):

        x_data = data[0]            # train data
        y_data = data[1]            # train label
        predicts = mnist(x_data)    # predict result

        # calculate loss
        loss = loss_fn(predicts, y_data)

        # calculate accuracy
        acc = paddle.metric.accuracy(predicts, y_data)

        # backpropagation
        loss.backward()

        if (batch_id+1) % 900 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id+1, loss.numpy(), acc.numpy()))

        # update parameters
        optim.step()

        # clear grad
        optim.clear_grad()
```

    epoch: 0, batch_id: 900, loss is: [0.09093174], acc is: [0.96875]
    epoch: 1, batch_id: 900, loss is: [0.00233106], acc is: [1.]
    epoch: 2, batch_id: 900, loss is: [0.04871231], acc is: [0.96875]
    epoch: 3, batch_id: 900, loss is: [0.00258381], acc is: [1.]
    epoch: 4, batch_id: 900, loss is: [0.06593578], acc is: [0.984375]


### 3.2 Evaluation with basic API

Evaluation with basic API，corresponding to Model.evaluate.


```python
# loading val dataset
test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, drop_last=True)
loss_fn = paddle.nn.CrossEntropyLoss()

mnist.eval()

for batch_id, data in enumerate(test_loader()):

    x_data = data[0]            # val data
    y_data = data[1]            # val label
    predicts = mnist(x_data)    # predict result

    # calculate loss and accuracy
    loss = loss_fn(predicts, y_data)
    acc = paddle.metric.accuracy(predicts, y_data)

    # print info
    if (batch_id+1) % 30 == 0:
        print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id+1, loss.numpy(), acc.numpy()))
```

    batch_id: 30, loss is: [0.15609288], acc is: [0.96875]
    batch_id: 60, loss is: [0.24816637], acc is: [0.953125]
    batch_id: 90, loss is: [0.06216685], acc is: [0.96875]
    batch_id: 120, loss is: [0.00033541], acc is: [1.]
    batch_id: 150, loss is: [0.11649689], acc is: [0.984375]


### 3.3 Prediction with basic API

Prediction with basic API，corresponding to Model.predict.


```python
# loading test dataset
test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, drop_last=True)

mnist.eval()
for batch_id, data in enumerate(test_loader()):
    x_data = data[0]
    predicts = mnist(x_data)
print("predict finished")
```

    predict finished

