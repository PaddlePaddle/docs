动态图
======

**作者:** `PaddlePaddle <https://github.com/PaddlePaddle>`__ **日期:**
2021.01 **摘要:**
从飞桨开源框架2.0版本开始，飞桨默认为用户开启了动态图开发模式。在这种模式下，每次执行一个运算，可以立即得到结果（而不是事先定义好网络结构，然后再执行）。在动态图模式下，你可以更加方便的组织代码，更容易的调试程序，本示例教程将向你介绍飞桨的动态图的使用。

一、环境配置
------------

本教程基于Paddle 2.0
编写，如果您的环境不是本版本，请先参考官网\ `安装 <https://www.paddlepaddle.org.cn/install/quick>`__
Paddle 2.0 。

.. code:: ipython3

    import paddle
    import paddle.nn.functional as F
    import numpy as np
    
    print(paddle.__version__)


.. parsed-literal::

    2.0.0


二、基本用法
------------

在动态图模式下，ni可以直接运行一个飞桨提供的API，它会立刻返回结果到python。不再需要首先创建一个计算图，然后再给定数据去运行。

.. code:: ipython3

    a = paddle.randn([4, 2])
    b = paddle.arange(1, 3, dtype='float32')
    
    print(a)
    print(b)
    
    c = a + b
    print(c)
    
    d = paddle.matmul(a, b)
    print(d)


.. parsed-literal::

    Tensor(shape=[4, 2], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[-0.69923532,  0.61731714],
            [-0.73845196, -0.32286811],
            [ 1.05555582,  0.83517230],
            [-0.54771936,  0.10178845]])
    Tensor(shape=[2], dtype=float32, place=CPUPlace, stop_gradient=True,
           [1., 2.])
    Tensor(shape=[4, 2], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[0.30076468, 2.61731720],
            [0.26154804, 1.67713189],
            [2.05555582, 2.83517218],
            [0.45228064, 2.10178852]])
    Tensor(shape=[4], dtype=float32, place=CPUPlace, stop_gradient=True,
           [ 0.53539896, -1.38418818,  2.72590041, -0.34414244])


三、使用python的控制流
----------------------

动态图模式下，您可以使用python的条件判断和循环，这类控制语句来执行神经网络的计算。（不再需要\ ``cond``,
``loop``\ 这类OP)

.. code:: ipython3

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



.. parsed-literal::

    0 +> [5 6 7]
    1 +> [5 7 9]
    2 +> [ 5  9 15]
    3 -> [-3  3 21]
    4 +> [ 5 21 87]
    5 -> [ -3  27 237]
    6 +> [  5  69 735]
    7 -> [  -3  123 2181]
    8 -> [  -3  251 6555]
    9 -> [   -3   507 19677]


四、构建更加灵活的网络：控制流
------------------------------

-  使用动态图可以用来创建更加灵活的网络，比如根据控制流选择不同的分支网络，和方便的构建权重共享的网络。接下来我们来看一个具体的例子，在这个例子中，第二个线性变换只有0.5的可能性会运行。
-  在sequence to sequence with
   attention的机器翻译的示例中，你会看到更实际的使用动态图构建RNN类的网络带来的灵活性。

.. code:: ipython3

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

.. code:: ipython3

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


.. parsed-literal::

    0 [1.3321278]
    200 [0.82869977]
    400 [0.57817966]
    600 [0.47205922]
    800 [0.15331703]
    1000 [0.06435855]
    1200 [0.07417449]
    1400 [0.04293861]
    1600 [0.02513926]
    1800 [0.02017618]
    2000 [0.00986554]
    2200 [0.00199787]
    2400 [0.00101295]
    2600 [0.00083713]
    2800 [0.00051033]


五、构建更加灵活的网络：共享权重
--------------------------------

-  使用动态图还可以更加方便的创建共享权重的网络，下面的示例展示了一个共享了权重的简单的AutoEncoder。
-  你也可以参考图像搜索的示例看到共享参数权重的更实际的使用。

.. code:: ipython3

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


.. parsed-literal::

    step: 0, loss: [0.33400834]
    step: 1, loss: [0.31674492]
    step: 2, loss: [0.29477125]
    step: 3, loss: [0.2680785]
    step: 4, loss: [0.23595281]
    step: 5, loss: [0.20035137]
    step: 6, loss: [0.16562223]
    step: 7, loss: [0.13548139]
    step: 8, loss: [0.11210174]
    step: 9, loss: [0.09638017]


The End
-------

可以看到使用动态图带来了更灵活易用的方式来组网和训练。你也可以在【使用注意力机制的LSTM的机器翻译】和【图片检索】两个示例中看到更完整的动态图的实际应用的灵活和便利。
