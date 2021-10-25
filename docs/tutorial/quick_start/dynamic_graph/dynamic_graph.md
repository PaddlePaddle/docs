# 动态图

**作者:** [PaddlePaddle](https://github.com/PaddlePaddle) <br>
**日期:** 2021.10 <br>
**摘要:** 从飞桨框架2.0版本开始，飞桨默认为开启了动态图开发模式。在这种模式下，每次执行一个运算，可以立即得到结果（而不是事先定义好网络结构，然后再执行）。在动态图模式下，你可以更加方便的组织代码，更容易的调试程序，本示例教程将向你介绍飞桨的动态图的使用。


## 一、环境配置

本教程基于 Paddle 2.2.0-rc0 编写，如果你的环境不是本版本，请先参考官网[安装](https://www.paddlepaddle.org.cn/install/quick) Paddle 2.2.0-rc0。


```python
import paddle
import paddle.nn.functional as F
import numpy as np

print(paddle.__version__)
```

    2.2.0-rc0


## 二、基本用法

在动态图模式下，你可以直接运行一个飞桨提供的API，它会立刻返回结果到python。不再需要首先创建一个计算图，然后再给定数据去运行。


```python
a = paddle.randn([4, 2])
b = paddle.arange(1, 3, dtype='float32')

print(a)
print(b)

c = a + b
print(c)

d = paddle.matmul(a, b)
print(d)
```

    Tensor(shape=[4, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
           [[-0.24989827,  0.66201270],
            [-1.58135796,  0.07232992],
            [-0.96044230,  0.31790161],
            [ 1.12384641,  0.66361523]])
    Tensor(shape=[2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
           [1., 2.])
    Tensor(shape=[4, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
           [[ 0.75010175,  2.66201258],
            [-0.58135796,  2.07233000],
            [ 0.03955770,  2.31790161],
            [ 2.12384653,  2.66361523]])
    Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
           [ 1.07412708, -1.43669808, -0.32463908,  2.45107698])


## 三、使用python的控制流

动态图模式下，你可以使用python的条件判断和循环，这类控制语句来执行神经网络的计算。（不再需要`cond`, `loop`这类OP)



```python
a = paddle.to_tensor(np.array([1, 2, 3]))
b = paddle.to_tensor(np.array([4, 5, 6]))

for i in range(10):
    r = paddle.rand([1,])
    if r > 0.5:
        c = paddle.pow(a, i) + b
        print("{} +> {}".format(i, c.numpy()))
    else:
        c = paddle.pow(a, i) - b
        print("{} -> {}".format(i, c.numpy()))

```

    0 +> [5 6 7]
    1 +> [5 7 9]
    2 -> [-3 -1  3]
    3 +> [ 5 13 33]
    4 -> [-3 11 75]
    5 -> [ -3  27 237]
    6 +> [  5  69 735]
    7 -> [  -3  123 2181]
    8 +> [   5  261 6567]
    9 -> [   -3   507 19677]


## 四、构建更加灵活的网络：控制流

- 使用动态图可以用来创建更加灵活的网络，比如根据控制流选择不同的分支网络，和方便的构建权重共享的网络。接下来来看一个具体的例子，在这个例子中，第二个线性变换只有0.5的可能性会运行。
- 在sequence to sequence with attention的机器翻译的示例中，你会看到更实际的使用动态图构建RNN类的网络带来的灵活性。



```python
class MyModel(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size):
        super(MyModel, self).__init__()
        self.linear1 = paddle.nn.Linear(input_size, hidden_size)
        self.linear2 = paddle.nn.Linear(hidden_size, hidden_size)
        self.linear3 = paddle.nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)

        if paddle.rand([1,]) > 0.5: 
            x = self.linear2(x)
            x = F.relu(x)

        x = self.linear3(x)
        
        return x     
```


```python
total_data, batch_size, input_size, hidden_size = 1000, 64, 128, 256

x_data = np.random.randn(total_data, input_size).astype(np.float32)
y_data = np.random.randn(total_data, 1).astype(np.float32)

model = MyModel(input_size, hidden_size)

loss_fn = paddle.nn.MSELoss(reduction='mean')
optimizer = paddle.optimizer.SGD(learning_rate=0.01, 
                                 parameters=model.parameters())

for t in range(200 * (total_data // batch_size)):
    idx = np.random.choice(total_data, batch_size, replace=False)
    x = paddle.to_tensor(x_data[idx,:])
    y = paddle.to_tensor(y_data[idx,:])
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    if t % 200 == 0:
        print(t, loss.numpy())

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
```

    0 [1.6115397]
    200 [0.62819237]
    400 [0.51455796]
    600 [0.20821849]
    800 [0.18254566]
    1000 [0.10306169]
    1200 [0.0135097]
    1400 [0.04026232]
    1600 [0.0045407]
    1800 [0.00335173]
    2000 [0.00926692]
    2200 [0.00407835]
    2400 [0.00521907]
    2600 [0.00268011]
    2800 [0.00156184]


## 五、构建更加灵活的网络：共享权重

- 使用动态图还可以更加方便的创建共享权重的网络，下面的示例展示了一个共享了权重的简单的AutoEncoder。
- 你也可以参考图像搜索的示例看到共享参数权重的更实际的使用。


```python
inputs = paddle.rand((256, 64))

linear = paddle.nn.Linear(64, 8, bias_attr=False)
loss_fn = paddle.nn.MSELoss()
optimizer = paddle.optimizer.Adam(0.01, parameters=linear.parameters())

for i in range(10):
    hidden = linear(inputs)
    # weight from input to hidden is shared with the linear mapping from hidden to output
    outputs = paddle.matmul(hidden, linear.weight, transpose_y=True) 
    loss = loss_fn(outputs, inputs)
    loss.backward()
    print("step: {}, loss: {}".format(i, loss.numpy()))
    optimizer.step()
    optimizer.clear_grad()
```

    step: 0, loss: [0.33082125]
    step: 1, loss: [0.29083017]
    step: 2, loss: [0.26794958]
    step: 3, loss: [0.24572413]
    step: 4, loss: [0.21951458]
    step: 5, loss: [0.19060956]
    step: 6, loss: [0.16274509]
    step: 7, loss: [0.13920584]
    step: 8, loss: [0.12127249]
    step: 9, loss: [0.1086377]


## The End

可以看到使用动态图带来了更灵活易用的方式来组网和训练。你也可以在【使用注意力机制的LSTM的机器翻译】和【图片检索】两个示例中看到更完整的动态图的实际应用的灵活和便利。
