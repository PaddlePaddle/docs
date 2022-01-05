# Saving and Loading Model

In this tutorial, it will be shown how to save and load the model.

## 1. Training Model before Saving

In general, you save the learned parameters of model.


```python
# Training

import paddle
from paddle.vision.transforms import ToTensor

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

class Mnist(paddle.nn.Layer):
    def __init__(self):
        super(Mnist, self).__init__()
        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(784, 512)
        self.linear_2 = paddle.nn.Linear(512, 10)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)

    def forward(self, inputs):
        y = self.flatten(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        return y

train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

mnist=Mnist()
mnist.train()

epochs = 5

optim = paddle.optimizer.Adam(parameters=mnist.parameters())

loss_fn = paddle.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):

        x_data = data[0]            
        y_data = data[1]            
        predicts = mnist(x_data)    

        loss = loss_fn(predicts, y_data)

        acc = paddle.metric.accuracy(predicts, y_data)

        loss.backward()

        if (batch_id+1) % 900 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id+1, loss.numpy(), acc.numpy()))

        optim.step()

        optim.clear_grad()
```

    epoch: 0, batch_id: 900, loss is: [0.10948133], acc is: [0.96875]
    epoch: 1, batch_id: 900, loss is: [0.11008529], acc is: [0.96875]
    epoch: 2, batch_id: 900, loss is: [0.03225169], acc is: [1.]
    epoch: 3, batch_id: 900, loss is: [0.01883681], acc is: [1.]
    epoch: 4, batch_id: 900, loss is: [0.06082885], acc is: [0.984375]


## 2. Saving Model

You can use `paddle.save` to save state_dict of Layer/Optimizer, Tensor and nested structure containing Tensor, Program. 


```python
paddle.save(mnist.state_dict(), "./mnist.pdparams")
paddle.save(optim.state_dict(), "./optim.pdopt")
```

## 3. Loading Model

You can use `paddle.load` to load state_dict of Layer/Optimizer, Tensor and nested structure containing Tensor, Program. 


```python
mnist_state_dict = paddle.load("./mnist.pdparams")
opt_state_dict = paddle.load("./optim.pdopt")

mnist.set_state_dict(mnist_state_dict)
optim.set_state_dict(opt_state_dict)
```
