MNIST数据集使用LeNet进行图像分类
================================

本示例教程演示如何在MNIST数据集上用LeNet进行图像分类。
手写数字的MNIST数据集，包含60,000个用于训练的示例和10,000个用于测试的示例。这些数字已经过尺寸标准化并位于图像中心，图像是固定大小(28x28像素)，其值为0到1。该数据集的官方地址为：http://yann.lecun.com/exdb/mnist/

环境
----

本教程基于paddle-develop编写，如果您的环境不是本版本，请先安装paddle-develop版本。

.. code:: ipython3

    import paddle
    print(paddle.__version__)
    paddle.disable_static()


.. parsed-literal::

    0.0.0


加载数据集
----------

我们使用飞桨自带的paddle.dataset完成mnist数据集的加载。

.. code:: ipython3

    print('download training data and load training data')
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    test_dataset = paddle.vision.datasets.MNIST(mode='test')
    print('load finished')


.. parsed-literal::

    /Library/Python/3.7/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


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



.. image:: https://github.com/PaddlePaddle/FluidDoc/blob/0717623bc28e74f527fae148a2814dd279aa7a7e/doc/paddle/tutorial/cv_case/mnist_lenet_classification/mnist_lenet_classification_files/mnist_lenet_classification_6_1.png
    

2.组网
------

用paddle.nn下的API，如\ ``Conv2d``\ 、\ ``Pool2D``\ 、\ ``Linead``\ 完成LeNet的构建。

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
            x = paddle.reshape(x, shape=[-1, 16*5*5])
            x = self.linear1(x)
            x = F.relu(x)
            x = self.linear2(x)
            x = F.relu(x)
            x = self.linear3(x)
            x = F.softmax(x)
            return x


.. parsed-literal::

    /Library/Python/3.7/site-packages/ipykernel/ipkernel.py:287: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)


3.训练方式一
------------

组网后，开始对模型进行训练，先构建\ ``train_loader``\ ，加载训练数据，然后定义\ ``train``\ 函数，设置好损失函数后，按batch加载数据，完成模型的训练。

.. code:: ipython3

    import paddle
    train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=64)
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
                optim.minimize(avg_loss)
                model.clear_gradients()
    model = LeNet()
    train(model)


.. parsed-literal::

    epoch: 0, batch_id: 0, loss is: [2.3062382], acc is: [0.109375]
    epoch: 0, batch_id: 100, loss is: [1.6826601], acc is: [0.84375]
    epoch: 0, batch_id: 200, loss is: [1.685574], acc is: [0.796875]
    epoch: 0, batch_id: 300, loss is: [1.5752499], acc is: [0.96875]
    epoch: 0, batch_id: 400, loss is: [1.5006541], acc is: [1.]
    epoch: 0, batch_id: 500, loss is: [1.5343401], acc is: [0.984375]
    epoch: 0, batch_id: 600, loss is: [1.4875913], acc is: [0.984375]
    epoch: 0, batch_id: 700, loss is: [1.5139006], acc is: [0.984375]
    epoch: 0, batch_id: 800, loss is: [1.5227785], acc is: [0.984375]
    epoch: 0, batch_id: 900, loss is: [1.4938308], acc is: [1.]
    epoch: 1, batch_id: 0, loss is: [1.4826943], acc is: [0.984375]
    epoch: 1, batch_id: 100, loss is: [1.4852213], acc is: [0.984375]
    epoch: 1, batch_id: 200, loss is: [1.5008337], acc is: [1.]
    epoch: 1, batch_id: 300, loss is: [1.505826], acc is: [1.]
    epoch: 1, batch_id: 400, loss is: [1.4768786], acc is: [1.]
    epoch: 1, batch_id: 500, loss is: [1.4950027], acc is: [0.984375]
    epoch: 1, batch_id: 600, loss is: [1.4762383], acc is: [0.984375]
    epoch: 1, batch_id: 700, loss is: [1.5276604], acc is: [0.96875]
    epoch: 1, batch_id: 800, loss is: [1.4897399], acc is: [1.]
    epoch: 1, batch_id: 900, loss is: [1.4927337], acc is: [1.]


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
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            # 获取预测结果
            loss = paddle.nn.functional.cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data, k=2)
            avg_loss = paddle.mean(loss)
            avg_acc = paddle.mean(acc)
            avg_loss.backward()
            if batch_id % 100 == 0:
                print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, avg_loss.numpy(), avg_acc.numpy()))
    test(model)


.. parsed-literal::

    batch_id: 0, loss is: [1.4630548], acc is: [1.]
    batch_id: 100, loss is: [1.4789999], acc is: [0.984375]
    batch_id: 200, loss is: [1.4621592], acc is: [1.]
    batch_id: 300, loss is: [1.486401], acc is: [1.]
    batch_id: 400, loss is: [1.4767764], acc is: [1.]
    batch_id: 500, loss is: [1.4987783], acc is: [0.984375]
    batch_id: 600, loss is: [1.4767168], acc is: [1.]
    batch_id: 700, loss is: [1.4876428], acc is: [0.984375]
    batch_id: 800, loss is: [1.4924926], acc is: [0.984375]
    batch_id: 900, loss is: [1.4799261], acc is: [1.]


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
            save_dir='mnist_checkpoint')


.. parsed-literal::

    Epoch 1/2
    step  10/938 - loss: 2.2252 - acc_top1: 0.2547 - acc_top2: 0.4234 - 16ms/step


.. parsed-literal::

    /Library/Python/3.7/site-packages/paddle/fluid/layers/utils.py:76: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return (isinstance(seq, collections.Sequence) and


.. parsed-literal::

    step  20/938 - loss: 1.9721 - acc_top1: 0.3664 - acc_top2: 0.5164 - 15ms/step
    step  30/938 - loss: 1.8697 - acc_top1: 0.4464 - acc_top2: 0.5651 - 15ms/step
    step  40/938 - loss: 1.8475 - acc_top1: 0.4859 - acc_top2: 0.5898 - 15ms/step
    step  50/938 - loss: 1.8683 - acc_top1: 0.5256 - acc_top2: 0.6156 - 14ms/step
    step  60/938 - loss: 1.8091 - acc_top1: 0.5437 - acc_top2: 0.6237 - 14ms/step
    step  70/938 - loss: 1.7934 - acc_top1: 0.5607 - acc_top2: 0.6335 - 14ms/step
    step  80/938 - loss: 1.7796 - acc_top1: 0.5760 - acc_top2: 0.6418 - 14ms/step
    step  90/938 - loss: 1.8004 - acc_top1: 0.5868 - acc_top2: 0.6476 - 14ms/step
    step 100/938 - loss: 1.7650 - acc_top1: 0.5972 - acc_top2: 0.6536 - 14ms/step
    step 110/938 - loss: 1.7839 - acc_top1: 0.6033 - acc_top2: 0.6570 - 14ms/step
    step 120/938 - loss: 1.8094 - acc_top1: 0.6087 - acc_top2: 0.6592 - 14ms/step
    step 130/938 - loss: 1.8125 - acc_top1: 0.6153 - acc_top2: 0.6638 - 14ms/step
    step 140/938 - loss: 1.7318 - acc_top1: 0.6217 - acc_top2: 0.6673 - 14ms/step
    step 150/938 - loss: 1.8209 - acc_top1: 0.6267 - acc_top2: 0.6702 - 14ms/step
    step 160/938 - loss: 1.7661 - acc_top1: 0.6308 - acc_top2: 0.6725 - 14ms/step
    step 170/938 - loss: 1.7099 - acc_top1: 0.6341 - acc_top2: 0.6741 - 14ms/step
    step 180/938 - loss: 1.8059 - acc_top1: 0.6363 - acc_top2: 0.6753 - 14ms/step
    step 190/938 - loss: 1.7681 - acc_top1: 0.6400 - acc_top2: 0.6779 - 14ms/step
    step 200/938 - loss: 1.8631 - acc_top1: 0.6430 - acc_top2: 0.6826 - 14ms/step
    step 210/938 - loss: 1.6808 - acc_top1: 0.6479 - acc_top2: 0.6879 - 14ms/step
    step 220/938 - loss: 1.5447 - acc_top1: 0.6558 - acc_top2: 0.6965 - 14ms/step
    step 230/938 - loss: 1.6170 - acc_top1: 0.6641 - acc_top2: 0.7051 - 14ms/step
    step 240/938 - loss: 1.6190 - acc_top1: 0.6719 - acc_top2: 0.7134 - 14ms/step
    step 250/938 - loss: 1.5698 - acc_top1: 0.6794 - acc_top2: 0.7209 - 14ms/step
    step 260/938 - loss: 1.6071 - acc_top1: 0.6869 - acc_top2: 0.7284 - 14ms/step
    step 270/938 - loss: 1.5507 - acc_top1: 0.6939 - acc_top2: 0.7364 - 14ms/step
    step 280/938 - loss: 1.5286 - acc_top1: 0.7023 - acc_top2: 0.7451 - 14ms/step
    step 290/938 - loss: 1.5740 - acc_top1: 0.7098 - acc_top2: 0.7532 - 14ms/step
    step 300/938 - loss: 1.5179 - acc_top1: 0.7172 - acc_top2: 0.7608 - 14ms/step
    step 310/938 - loss: 1.5325 - acc_top1: 0.7240 - acc_top2: 0.7677 - 14ms/step
    step 320/938 - loss: 1.4961 - acc_top1: 0.7305 - acc_top2: 0.7744 - 14ms/step
    step 330/938 - loss: 1.5420 - acc_top1: 0.7369 - acc_top2: 0.7804 - 14ms/step
    step 340/938 - loss: 1.5652 - acc_top1: 0.7427 - acc_top2: 0.7861 - 14ms/step
    step 350/938 - loss: 1.5122 - acc_top1: 0.7484 - acc_top2: 0.7918 - 14ms/step
    step 360/938 - loss: 1.5308 - acc_top1: 0.7544 - acc_top2: 0.7972 - 14ms/step
    step 370/938 - loss: 1.5354 - acc_top1: 0.7596 - acc_top2: 0.8023 - 14ms/step
    step 380/938 - loss: 1.5433 - acc_top1: 0.7645 - acc_top2: 0.8073 - 14ms/step
    step 390/938 - loss: 1.5341 - acc_top1: 0.7693 - acc_top2: 0.8119 - 14ms/step
    step 400/938 - loss: 1.4826 - acc_top1: 0.7740 - acc_top2: 0.8163 - 14ms/step
    step 410/938 - loss: 1.4995 - acc_top1: 0.7785 - acc_top2: 0.8205 - 14ms/step
    step 420/938 - loss: 1.5057 - acc_top1: 0.7827 - acc_top2: 0.8244 - 14ms/step
    step 430/938 - loss: 1.4927 - acc_top1: 0.7866 - acc_top2: 0.8282 - 14ms/step
    step 440/938 - loss: 1.5281 - acc_top1: 0.7902 - acc_top2: 0.8316 - 14ms/step
    step 450/938 - loss: 1.5060 - acc_top1: 0.7936 - acc_top2: 0.8347 - 14ms/step
    step 460/938 - loss: 1.5135 - acc_top1: 0.7968 - acc_top2: 0.8380 - 14ms/step
    step 470/938 - loss: 1.5206 - acc_top1: 0.8004 - acc_top2: 0.8411 - 14ms/step
    step 480/938 - loss: 1.4963 - acc_top1: 0.8039 - acc_top2: 0.8441 - 14ms/step
    step 490/938 - loss: 1.4984 - acc_top1: 0.8071 - acc_top2: 0.8470 - 14ms/step
    step 500/938 - loss: 1.4947 - acc_top1: 0.8101 - acc_top2: 0.8498 - 14ms/step
    step 510/938 - loss: 1.4639 - acc_top1: 0.8130 - acc_top2: 0.8524 - 14ms/step
    step 520/938 - loss: 1.4781 - acc_top1: 0.8158 - acc_top2: 0.8549 - 14ms/step
    step 530/938 - loss: 1.4806 - acc_top1: 0.8187 - acc_top2: 0.8575 - 14ms/step
    step 540/938 - loss: 1.4830 - acc_top1: 0.8214 - acc_top2: 0.8600 - 14ms/step
    step 550/938 - loss: 1.4852 - acc_top1: 0.8239 - acc_top2: 0.8623 - 14ms/step
    step 560/938 - loss: 1.5302 - acc_top1: 0.8263 - acc_top2: 0.8645 - 14ms/step
    step 570/938 - loss: 1.5520 - acc_top1: 0.8286 - acc_top2: 0.8667 - 14ms/step
    step 580/938 - loss: 1.4897 - acc_top1: 0.8305 - acc_top2: 0.8687 - 14ms/step
    step 590/938 - loss: 1.4857 - acc_top1: 0.8328 - acc_top2: 0.8707 - 14ms/step
    step 600/938 - loss: 1.5081 - acc_top1: 0.8351 - acc_top2: 0.8727 - 14ms/step
    step 610/938 - loss: 1.5013 - acc_top1: 0.8373 - acc_top2: 0.8746 - 14ms/step
    step 620/938 - loss: 1.4949 - acc_top1: 0.8395 - acc_top2: 0.8764 - 14ms/step
    step 630/938 - loss: 1.4971 - acc_top1: 0.8412 - acc_top2: 0.8781 - 14ms/step
    step 640/938 - loss: 1.4869 - acc_top1: 0.8434 - acc_top2: 0.8800 - 14ms/step
    step 650/938 - loss: 1.5202 - acc_top1: 0.8450 - acc_top2: 0.8815 - 14ms/step
    step 660/938 - loss: 1.5002 - acc_top1: 0.8468 - acc_top2: 0.8832 - 14ms/step
    step 670/938 - loss: 1.5178 - acc_top1: 0.8487 - acc_top2: 0.8848 - 14ms/step
    step 680/938 - loss: 1.4939 - acc_top1: 0.8504 - acc_top2: 0.8864 - 14ms/step
    step 690/938 - loss: 1.4650 - acc_top1: 0.8520 - acc_top2: 0.8878 - 14ms/step
    step 700/938 - loss: 1.4934 - acc_top1: 0.8537 - acc_top2: 0.8892 - 14ms/step
    step 710/938 - loss: 1.5473 - acc_top1: 0.8552 - acc_top2: 0.8905 - 14ms/step
    step 720/938 - loss: 1.4956 - acc_top1: 0.8568 - acc_top2: 0.8918 - 14ms/step
    step 730/938 - loss: 1.4644 - acc_top1: 0.8583 - acc_top2: 0.8932 - 14ms/step
    step 740/938 - loss: 1.4868 - acc_top1: 0.8598 - acc_top2: 0.8946 - 14ms/step
    step 750/938 - loss: 1.5142 - acc_top1: 0.8613 - acc_top2: 0.8959 - 14ms/step
    step 760/938 - loss: 1.4656 - acc_top1: 0.8628 - acc_top2: 0.8971 - 14ms/step
    step 770/938 - loss: 1.5005 - acc_top1: 0.8641 - acc_top2: 0.8983 - 14ms/step
    step 780/938 - loss: 1.5557 - acc_top1: 0.8653 - acc_top2: 0.8994 - 14ms/step
    step 790/938 - loss: 1.4687 - acc_top1: 0.8666 - acc_top2: 0.9006 - 14ms/step
    step 800/938 - loss: 1.4686 - acc_top1: 0.8680 - acc_top2: 0.9017 - 14ms/step
    step 810/938 - loss: 1.5202 - acc_top1: 0.8693 - acc_top2: 0.9028 - 14ms/step
    step 820/938 - loss: 1.4773 - acc_top1: 0.8705 - acc_top2: 0.9038 - 14ms/step
    step 830/938 - loss: 1.4838 - acc_top1: 0.8717 - acc_top2: 0.9049 - 14ms/step
    step 840/938 - loss: 1.4726 - acc_top1: 0.8728 - acc_top2: 0.9059 - 14ms/step
    step 850/938 - loss: 1.4734 - acc_top1: 0.8741 - acc_top2: 0.9069 - 14ms/step
    step 860/938 - loss: 1.4627 - acc_top1: 0.8752 - acc_top2: 0.9078 - 14ms/step
    step 870/938 - loss: 1.4872 - acc_top1: 0.8763 - acc_top2: 0.9088 - 14ms/step
    step 880/938 - loss: 1.4916 - acc_top1: 0.8773 - acc_top2: 0.9096 - 14ms/step
    step 890/938 - loss: 1.4818 - acc_top1: 0.8784 - acc_top2: 0.9105 - 14ms/step
    step 900/938 - loss: 1.4967 - acc_top1: 0.8794 - acc_top2: 0.9114 - 14ms/step
    step 910/938 - loss: 1.4614 - acc_top1: 0.8804 - acc_top2: 0.9123 - 14ms/step
    step 920/938 - loss: 1.4819 - acc_top1: 0.8815 - acc_top2: 0.9132 - 14ms/step
    step 930/938 - loss: 1.5114 - acc_top1: 0.8824 - acc_top2: 0.9140 - 14ms/step
    step 938/938 - loss: 1.4621 - acc_top1: 0.8832 - acc_top2: 0.9146 - 14ms/step
    save checkpoint at /Users/chenlong/online_repo/book/paddle2.0_docs/image_classification/mnist_checkpoint/0
    Epoch 2/2
    step  10/938 - loss: 1.5033 - acc_top1: 0.9734 - acc_top2: 0.9906 - 15ms/step
    step  20/938 - loss: 1.4812 - acc_top1: 0.9734 - acc_top2: 0.9906 - 14ms/step
    step  30/938 - loss: 1.4623 - acc_top1: 0.9714 - acc_top2: 0.9911 - 14ms/step
    step  40/938 - loss: 1.4775 - acc_top1: 0.9711 - acc_top2: 0.9918 - 14ms/step
    step  50/938 - loss: 1.4857 - acc_top1: 0.9712 - acc_top2: 0.9922 - 14ms/step
    step  60/938 - loss: 1.4895 - acc_top1: 0.9695 - acc_top2: 0.9904 - 14ms/step
    step  70/938 - loss: 1.4746 - acc_top1: 0.9708 - acc_top2: 0.9908 - 14ms/step
    step  80/938 - loss: 1.4945 - acc_top1: 0.9719 - acc_top2: 0.9912 - 14ms/step
    step  90/938 - loss: 1.4644 - acc_top1: 0.9722 - acc_top2: 0.9911 - 14ms/step
    step 100/938 - loss: 1.4727 - acc_top1: 0.9722 - acc_top2: 0.9912 - 14ms/step
    step 110/938 - loss: 1.4634 - acc_top1: 0.9720 - acc_top2: 0.9915 - 14ms/step
    step 120/938 - loss: 1.4856 - acc_top1: 0.9730 - acc_top2: 0.9915 - 14ms/step
    step 130/938 - loss: 1.4778 - acc_top1: 0.9736 - acc_top2: 0.9916 - 14ms/step
    step 140/938 - loss: 1.4949 - acc_top1: 0.9730 - acc_top2: 0.9914 - 14ms/step
    step 150/938 - loss: 1.4836 - acc_top1: 0.9726 - acc_top2: 0.9914 - 14ms/step
    step 160/938 - loss: 1.5430 - acc_top1: 0.9725 - acc_top2: 0.9917 - 14ms/step
    step 170/938 - loss: 1.4882 - acc_top1: 0.9722 - acc_top2: 0.9916 - 14ms/step
    step 180/938 - loss: 1.4777 - acc_top1: 0.9721 - acc_top2: 0.9919 - 14ms/step
    step 190/938 - loss: 1.4816 - acc_top1: 0.9723 - acc_top2: 0.9920 - 14ms/step
    step 200/938 - loss: 1.4916 - acc_top1: 0.9730 - acc_top2: 0.9923 - 14ms/step
    step 210/938 - loss: 1.5290 - acc_top1: 0.9734 - acc_top2: 0.9923 - 14ms/step
    step 220/938 - loss: 1.5006 - acc_top1: 0.9736 - acc_top2: 0.9923 - 14ms/step
    step 230/938 - loss: 1.5103 - acc_top1: 0.9737 - acc_top2: 0.9923 - 14ms/step
    step 240/938 - loss: 1.4905 - acc_top1: 0.9733 - acc_top2: 0.9920 - 14ms/step
    step 250/938 - loss: 1.5066 - acc_top1: 0.9734 - acc_top2: 0.9920 - 14ms/step
    step 260/938 - loss: 1.4846 - acc_top1: 0.9736 - acc_top2: 0.9920 - 14ms/step
    step 270/938 - loss: 1.4717 - acc_top1: 0.9738 - acc_top2: 0.9921 - 14ms/step
    step 280/938 - loss: 1.4648 - acc_top1: 0.9742 - acc_top2: 0.9921 - 14ms/step
    step 290/938 - loss: 1.4657 - acc_top1: 0.9745 - acc_top2: 0.9921 - 14ms/step
    step 300/938 - loss: 1.4630 - acc_top1: 0.9744 - acc_top2: 0.9920 - 14ms/step
    step 310/938 - loss: 1.5053 - acc_top1: 0.9742 - acc_top2: 0.9918 - 14ms/step
    step 320/938 - loss: 1.4843 - acc_top1: 0.9745 - acc_top2: 0.9919 - 14ms/step
    step 330/938 - loss: 1.4915 - acc_top1: 0.9745 - acc_top2: 0.9919 - 14ms/step
    step 340/938 - loss: 1.5146 - acc_top1: 0.9745 - acc_top2: 0.9918 - 14ms/step
    step 350/938 - loss: 1.4768 - acc_top1: 0.9742 - acc_top2: 0.9916 - 14ms/step
    step 360/938 - loss: 1.4827 - acc_top1: 0.9743 - acc_top2: 0.9918 - 14ms/step
    step 370/938 - loss: 1.5097 - acc_top1: 0.9740 - acc_top2: 0.9917 - 14ms/step
    step 380/938 - loss: 1.5225 - acc_top1: 0.9739 - acc_top2: 0.9916 - 14ms/step
    step 390/938 - loss: 1.4701 - acc_top1: 0.9740 - acc_top2: 0.9917 - 14ms/step
    step 400/938 - loss: 1.4986 - acc_top1: 0.9741 - acc_top2: 0.9920 - 14ms/step
    step 410/938 - loss: 1.5210 - acc_top1: 0.9740 - acc_top2: 0.9918 - 14ms/step
    step 420/938 - loss: 1.4799 - acc_top1: 0.9740 - acc_top2: 0.9917 - 14ms/step
    step 430/938 - loss: 1.4845 - acc_top1: 0.9744 - acc_top2: 0.9919 - 14ms/step
    step 440/938 - loss: 1.4773 - acc_top1: 0.9741 - acc_top2: 0.9918 - 14ms/step
    step 450/938 - loss: 1.4719 - acc_top1: 0.9743 - acc_top2: 0.9918 - 14ms/step
    step 460/938 - loss: 1.4773 - acc_top1: 0.9742 - acc_top2: 0.9918 - 14ms/step
    step 470/938 - loss: 1.4944 - acc_top1: 0.9741 - acc_top2: 0.9918 - 14ms/step
    step 480/938 - loss: 1.4793 - acc_top1: 0.9743 - acc_top2: 0.9919 - 14ms/step
    step 490/938 - loss: 1.4625 - acc_top1: 0.9746 - acc_top2: 0.9920 - 14ms/step
    step 500/938 - loss: 1.4829 - acc_top1: 0.9745 - acc_top2: 0.9921 - 14ms/step
    step 510/938 - loss: 1.4659 - acc_top1: 0.9747 - acc_top2: 0.9921 - 14ms/step
    step 520/938 - loss: 1.4862 - acc_top1: 0.9743 - acc_top2: 0.9921 - 14ms/step
    step 530/938 - loss: 1.5039 - acc_top1: 0.9742 - acc_top2: 0.9921 - 14ms/step
    step 540/938 - loss: 1.5070 - acc_top1: 0.9740 - acc_top2: 0.9921 - 14ms/step
    step 550/938 - loss: 1.5033 - acc_top1: 0.9740 - acc_top2: 0.9922 - 14ms/step
    step 560/938 - loss: 1.4846 - acc_top1: 0.9741 - acc_top2: 0.9921 - 14ms/step
    step 570/938 - loss: 1.4613 - acc_top1: 0.9741 - acc_top2: 0.9921 - 14ms/step
    step 580/938 - loss: 1.4616 - acc_top1: 0.9743 - acc_top2: 0.9921 - 14ms/step
    step 590/938 - loss: 1.4801 - acc_top1: 0.9745 - acc_top2: 0.9921 - 14ms/step
    step 600/938 - loss: 1.4772 - acc_top1: 0.9746 - acc_top2: 0.9921 - 14ms/step
    step 610/938 - loss: 1.4612 - acc_top1: 0.9746 - acc_top2: 0.9921 - 14ms/step
    step 620/938 - loss: 1.4951 - acc_top1: 0.9746 - acc_top2: 0.9922 - 14ms/step
    step 630/938 - loss: 1.4755 - acc_top1: 0.9747 - acc_top2: 0.9923 - 14ms/step
    step 640/938 - loss: 1.5296 - acc_top1: 0.9749 - acc_top2: 0.9924 - 14ms/step
    step 650/938 - loss: 1.5054 - acc_top1: 0.9748 - acc_top2: 0.9924 - 14ms/step
    step 660/938 - loss: 1.4775 - acc_top1: 0.9749 - acc_top2: 0.9925 - 14ms/step
    step 670/938 - loss: 1.4829 - acc_top1: 0.9749 - acc_top2: 0.9925 - 14ms/step
    step 680/938 - loss: 1.4612 - acc_top1: 0.9750 - acc_top2: 0.9926 - 14ms/step
    step 690/938 - loss: 1.4869 - acc_top1: 0.9751 - acc_top2: 0.9926 - 14ms/step
    step 700/938 - loss: 1.4612 - acc_top1: 0.9752 - acc_top2: 0.9927 - 14ms/step
    step 710/938 - loss: 1.5235 - acc_top1: 0.9752 - acc_top2: 0.9927 - 14ms/step
    step 720/938 - loss: 1.5317 - acc_top1: 0.9752 - acc_top2: 0.9926 - 14ms/step
    step 730/938 - loss: 1.4898 - acc_top1: 0.9751 - acc_top2: 0.9926 - 14ms/step
    step 740/938 - loss: 1.4612 - acc_top1: 0.9753 - acc_top2: 0.9926 - 14ms/step
    step 750/938 - loss: 1.4935 - acc_top1: 0.9752 - acc_top2: 0.9926 - 14ms/step
    step 760/938 - loss: 1.5140 - acc_top1: 0.9749 - acc_top2: 0.9926 - 14ms/step
    step 770/938 - loss: 1.4883 - acc_top1: 0.9748 - acc_top2: 0.9925 - 14ms/step
    step 780/938 - loss: 1.4759 - acc_top1: 0.9748 - acc_top2: 0.9926 - 14ms/step
    step 790/938 - loss: 1.4773 - acc_top1: 0.9750 - acc_top2: 0.9926 - 14ms/step
    step 800/938 - loss: 1.4766 - acc_top1: 0.9750 - acc_top2: 0.9926 - 14ms/step
    step 810/938 - loss: 1.5058 - acc_top1: 0.9750 - acc_top2: 0.9927 - 14ms/step
    step 820/938 - loss: 1.4867 - acc_top1: 0.9749 - acc_top2: 0.9927 - 14ms/step
    step 830/938 - loss: 1.4766 - acc_top1: 0.9748 - acc_top2: 0.9927 - 14ms/step
    step 840/938 - loss: 1.4680 - acc_top1: 0.9747 - acc_top2: 0.9927 - 14ms/step
    step 850/938 - loss: 1.4628 - acc_top1: 0.9746 - acc_top2: 0.9927 - 14ms/step
    step 860/938 - loss: 1.5035 - acc_top1: 0.9747 - acc_top2: 0.9928 - 14ms/step
    step 870/938 - loss: 1.4857 - acc_top1: 0.9748 - acc_top2: 0.9928 - 14ms/step
    step 880/938 - loss: 1.4767 - acc_top1: 0.9748 - acc_top2: 0.9927 - 14ms/step
    step 890/938 - loss: 1.4612 - acc_top1: 0.9750 - acc_top2: 0.9928 - 14ms/step
    step 900/938 - loss: 1.4620 - acc_top1: 0.9751 - acc_top2: 0.9928 - 14ms/step
    step 910/938 - loss: 1.4621 - acc_top1: 0.9751 - acc_top2: 0.9928 - 14ms/step
    step 920/938 - loss: 1.4768 - acc_top1: 0.9751 - acc_top2: 0.9927 - 14ms/step
    step 930/938 - loss: 1.4806 - acc_top1: 0.9752 - acc_top2: 0.9928 - 14ms/step
    step 938/938 - loss: 1.4910 - acc_top1: 0.9752 - acc_top2: 0.9928 - 14ms/step
    save checkpoint at /Users/chenlong/online_repo/book/paddle2.0_docs/image_classification/mnist_checkpoint/1
    save checkpoint at /Users/chenlong/online_repo/book/paddle2.0_docs/image_classification/mnist_checkpoint/final


使用model.evaluate来预测模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    model.evaluate(test_dataset, batch_size=64)


.. parsed-literal::

    Eval begin...
    step  10/157 - loss: 1.5014 - acc_top1: 0.9766 - acc_top2: 0.9953 - 6ms/step
    step  20/157 - loss: 1.5239 - acc_top1: 0.9742 - acc_top2: 0.9922 - 6ms/step
    step  30/157 - loss: 1.4926 - acc_top1: 0.9740 - acc_top2: 0.9932 - 6ms/step
    step  40/157 - loss: 1.4612 - acc_top1: 0.9734 - acc_top2: 0.9938 - 6ms/step
    step  50/157 - loss: 1.4612 - acc_top1: 0.9719 - acc_top2: 0.9938 - 6ms/step
    step  60/157 - loss: 1.5114 - acc_top1: 0.9721 - acc_top2: 0.9938 - 6ms/step
    step  70/157 - loss: 1.4793 - acc_top1: 0.9696 - acc_top2: 0.9935 - 6ms/step
    step  80/157 - loss: 1.4736 - acc_top1: 0.9695 - acc_top2: 0.9932 - 6ms/step
    step  90/157 - loss: 1.4892 - acc_top1: 0.9720 - acc_top2: 0.9939 - 6ms/step
    step 100/157 - loss: 1.4623 - acc_top1: 0.9738 - acc_top2: 0.9941 - 6ms/step
    step 110/157 - loss: 1.4612 - acc_top1: 0.9737 - acc_top2: 0.9939 - 6ms/step
    step 120/157 - loss: 1.4612 - acc_top1: 0.9746 - acc_top2: 0.9939 - 6ms/step
    step 130/157 - loss: 1.4703 - acc_top1: 0.9757 - acc_top2: 0.9942 - 6ms/step
    step 140/157 - loss: 1.4612 - acc_top1: 0.9771 - acc_top2: 0.9946 - 6ms/step
    step 150/157 - loss: 1.4748 - acc_top1: 0.9782 - acc_top2: 0.9950 - 6ms/step
    step 157/157 - loss: 1.4612 - acc_top1: 0.9770 - acc_top2: 0.9949 - 6ms/step
    Eval samples: 10000




.. parsed-literal::

    {'loss': [1.4611504], 'acc_top1': 0.977, 'acc_top2': 0.9949}



训练方式二结束
~~~~~~~~~~~~~~

以上就是训练方式二，可以快速、高效的完成网络模型训练与预测。

总结
----

以上就是用LeNet对手写数字数据及MNIST进行分类。本示例提供了两种训练模型的方式，一种可以快速完成模型的组建与预测，非常适合新手用户上手。另一种则需要多个步骤来完成模型的训练，适合进阶用户使用。
