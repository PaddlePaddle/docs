MNIST数据集使用LeNet进行图像分类
================================

本示例教程演示如何在MNIST数据集上用LeNet进行图像分类。
手写数字的MNIST数据集，包含60,000个用于训练的示例和10,000个用于测试的示例。这些数字已经过尺寸标准化并位于图像中心，图像是固定大小(28x28像素)，其值为0到1。该数据集的官方地址为：http://yann.lecun.com/exdb/mnist/

环境
----

本教程基于paddle-2.0-beta编写，如果您的环境不是本版本，请先安装paddle-2.0-beta版本。

.. code:: ipython3

    import paddle
    print(paddle.__version__)
    paddle.disable_static()
    # 开启动态图


.. parsed-literal::

    2.0.0-beta0


加载数据集
----------

我们使用飞桨自带的paddle.dataset完成mnist数据集的加载。

.. code:: ipython3

    print('download training data and load training data')
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    test_dataset = paddle.vision.datasets.MNIST(mode='test')
    print('load finished')


.. parsed-literal::

    download training data and load training data
    load finished


取训练集中的一条数据看一下。

.. code:: ipython3

    import numpy as np
    import matplotlib.pyplot as plt
    train_data0, train_label_0 = train_dataset[0][0],train_dataset[0][1]
    train_data0 = train_data0.reshape([28,28])
    plt.figure(figsize=(2,2))
    plt.imshow(train_data0, cmap=plt.cm.binary)
    print('train_data0 label is: ' + str(train_label_0))


.. parsed-literal::

    train_data0 label is: [5]



.. image:: https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/tutorial/cv_case/mnist_lenet_classification/mnist_lenet_classification_files/mnist_lenet_classification_001.png?raw=true


组网
----

用paddle.nn下的API，如\ ``Conv2d``\ 、\ ``MaxPool2d``\ 、\ ``Linear``\ 完成LeNet的构建。

.. code:: ipython3

    import paddle
    import paddle.nn.functional as F
    class LeNet(paddle.nn.Layer):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = paddle.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
            self.max_pool1 = paddle.nn.MaxPool2d(kernel_size=2,  stride=2)
            self.conv2 = paddle.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
            self.max_pool2 = paddle.nn.MaxPool2d(kernel_size=2, stride=2)
            self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
            self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
            self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)
    
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.max_pool1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = self.max_pool2(x)
            x = paddle.flatten(x, start_axis=1,stop_axis=-1)
            x = self.linear1(x)
            x = F.relu(x)
            x = self.linear2(x)
            x = F.relu(x)
            x = self.linear3(x)
            x = F.softmax(x)
            return x

训练方式一
----------

组网后，开始对模型进行训练，先构建\ ``train_loader``\ ，加载训练数据，然后定义\ ``train``\ 函数，设置好损失函数后，按batch加载数据，完成模型的训练。

.. code:: ipython3

    import paddle
    train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=64, shuffle=True)
    # 加载训练集 batch_size 设为 64
    def train(model):
        model.train()
        epochs = 2
        optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
        # 用Adam作为优化函数
        for epoch in range(epochs):
            for batch_id, data in enumerate(train_loader()):
                x_data = data[0]
                y_data = data[1]
                predicts = model(x_data)
                loss = paddle.nn.functional.cross_entropy(predicts, y_data)
                # 计算损失
                acc = paddle.metric.accuracy(predicts, y_data, k=2)
                avg_loss = paddle.mean(loss)
                avg_acc = paddle.mean(acc)
                avg_loss.backward()
                if batch_id % 100 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, avg_loss.numpy(), avg_acc.numpy()))
                optim.step()
                optim.clear_grad()
    model = LeNet()
    train(model)


.. parsed-literal::

    epoch: 0, batch_id: 0, loss is: [2.3037894], acc is: [0.140625]
    epoch: 0, batch_id: 100, loss is: [1.6175328], acc is: [0.9375]
    epoch: 0, batch_id: 200, loss is: [1.5388051], acc is: [0.96875]
    epoch: 0, batch_id: 300, loss is: [1.5251061], acc is: [0.96875]
    epoch: 0, batch_id: 400, loss is: [1.4678856], acc is: [1.]
    epoch: 0, batch_id: 500, loss is: [1.4944503], acc is: [0.984375]
    epoch: 0, batch_id: 600, loss is: [1.5365536], acc is: [0.96875]
    epoch: 0, batch_id: 700, loss is: [1.4885054], acc is: [0.984375]
    epoch: 0, batch_id: 800, loss is: [1.4872254], acc is: [0.984375]
    epoch: 0, batch_id: 900, loss is: [1.4884174], acc is: [0.984375]
    epoch: 1, batch_id: 0, loss is: [1.4776722], acc is: [1.]
    epoch: 1, batch_id: 100, loss is: [1.4751343], acc is: [1.]
    epoch: 1, batch_id: 200, loss is: [1.4772581], acc is: [1.]
    epoch: 1, batch_id: 300, loss is: [1.4918218], acc is: [0.984375]
    epoch: 1, batch_id: 400, loss is: [1.5038397], acc is: [0.96875]
    epoch: 1, batch_id: 500, loss is: [1.5088196], acc is: [0.96875]
    epoch: 1, batch_id: 600, loss is: [1.4961376], acc is: [0.984375]
    epoch: 1, batch_id: 700, loss is: [1.4755756], acc is: [1.]
    epoch: 1, batch_id: 800, loss is: [1.4921497], acc is: [0.984375]
    epoch: 1, batch_id: 900, loss is: [1.4944404], acc is: [1.]


对模型进行验证
~~~~~~~~~~~~~~

训练完成后，需要验证模型的效果，此时，加载测试数据集，然后用训练好的模对测试集进行预测，计算损失与精度。

.. code:: ipython3

    import paddle
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
    # 加载测试数据集
    def test(model):
        model.eval()
        batch_size = 64
        for batch_id, data in enumerate(test_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            # 获取预测结果
            loss = paddle.nn.functional.cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data, k=2)
            avg_loss = paddle.mean(loss)
            avg_acc = paddle.mean(acc)
            avg_loss.backward()
            if batch_id % 20 == 0:
                print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, avg_loss.numpy(), avg_acc.numpy()))
    test(model)


.. parsed-literal::

    batch_id: 0, loss is: [1.4915928], acc is: [1.]
    batch_id: 20, loss is: [1.4818308], acc is: [1.]
    batch_id: 40, loss is: [1.5006062], acc is: [0.984375]
    batch_id: 60, loss is: [1.521233], acc is: [1.]
    batch_id: 80, loss is: [1.4772738], acc is: [1.]
    batch_id: 100, loss is: [1.4755945], acc is: [1.]
    batch_id: 120, loss is: [1.4746133], acc is: [1.]
    batch_id: 140, loss is: [1.4786345], acc is: [1.]


训练方式一结束
~~~~~~~~~~~~~~

以上就是训练方式一，通过这种方式，可以清楚的看到训练和测试中的每一步过程。但是，这种方式句法比较复杂。因此，我们提供了训练方式二，能够更加快速、高效的完成模型的训练与测试。

3.训练方式二
------------

通过paddle提供的\ ``Model``
构建实例，使用封装好的训练与测试接口，快速完成模型训练与测试。

.. code:: ipython3

    import paddle
    from paddle.static import InputSpec
    from paddle.metric import Accuracy
    inputs = InputSpec([None, 784], 'float32', 'x')
    labels = InputSpec([None, 10], 'float32', 'x')
    model = paddle.Model(LeNet(), inputs, labels)
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    
    model.prepare(
        optim,
        paddle.nn.loss.CrossEntropyLoss(),
        Accuracy(topk=(1, 2))
        )

使用model.fit来训练模型
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    model.fit(train_dataset,
            epochs=2,
            batch_size=64,
            log_freq=200
            )


.. parsed-literal::

    Epoch 1/2
    step 200/938 - loss: 1.5219 - acc_top1: 0.9829 - acc_top2: 0.9965 - 14ms/step
    step 400/938 - loss: 1.4765 - acc_top1: 0.9825 - acc_top2: 0.9958 - 13ms/step
    step 600/938 - loss: 1.4624 - acc_top1: 0.9823 - acc_top2: 0.9953 - 13ms/step
    step 800/938 - loss: 1.4768 - acc_top1: 0.9829 - acc_top2: 0.9955 - 13ms/step
    step 938/938 - loss: 1.4612 - acc_top1: 0.9836 - acc_top2: 0.9956 - 13ms/step
    Epoch 2/2
    step 200/938 - loss: 1.4705 - acc_top1: 0.9834 - acc_top2: 0.9959 - 13ms/step
    step 400/938 - loss: 1.4620 - acc_top1: 0.9833 - acc_top2: 0.9960 - 13ms/step
    step 600/938 - loss: 1.4613 - acc_top1: 0.9830 - acc_top2: 0.9960 - 13ms/step
    step 800/938 - loss: 1.4763 - acc_top1: 0.9831 - acc_top2: 0.9960 - 13ms/step
    step 938/938 - loss: 1.4924 - acc_top1: 0.9834 - acc_top2: 0.9959 - 13ms/step


使用model.evaluate来预测模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    model.evaluate(test_dataset, log_freq=20, batch_size=64)


.. parsed-literal::

    Eval begin...
    step  20/157 - loss: 1.5246 - acc_top1: 0.9773 - acc_top2: 0.9969 - 6ms/step
    step  40/157 - loss: 1.4622 - acc_top1: 0.9758 - acc_top2: 0.9961 - 6ms/step
    step  60/157 - loss: 1.5241 - acc_top1: 0.9763 - acc_top2: 0.9951 - 6ms/step
    step  80/157 - loss: 1.4612 - acc_top1: 0.9787 - acc_top2: 0.9959 - 6ms/step
    step 100/157 - loss: 1.4612 - acc_top1: 0.9823 - acc_top2: 0.9967 - 5ms/step
    step 120/157 - loss: 1.4612 - acc_top1: 0.9835 - acc_top2: 0.9966 - 5ms/step
    step 140/157 - loss: 1.4612 - acc_top1: 0.9844 - acc_top2: 0.9969 - 5ms/step
    step 157/157 - loss: 1.4612 - acc_top1: 0.9838 - acc_top2: 0.9966 - 5ms/step
    Eval samples: 10000




.. parsed-literal::

    {'loss': [1.4611504], 'acc_top1': 0.9838, 'acc_top2': 0.9966}



训练方式二结束
~~~~~~~~~~~~~~

以上就是训练方式二，可以快速、高效的完成网络模型训练与预测。

总结
----

以上就是用LeNet对手写数字数据及MNIST进行分类。本示例提供了两种训练模型的方式，一种可以快速完成模型的组建与预测，非常适合新手用户上手。另一种则需要多个步骤来完成模型的训练，适合进阶用户使用。
