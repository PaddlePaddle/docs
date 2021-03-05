.. _cn_doc_quick_start:

10分钟快速上手飞桨（PaddlePaddle）
=================================

本示例通过一个基础案例，带你快速了解如何使用飞桨框架。

一、安装飞桨
-----------

如果你已经安装好飞桨那么可以跳过此步骤。飞桨提供了一个方便易用的安装引导页面，你可以通过选择自己的系统和软件版本来获取对应的安装命令，具体可以点击\ `快速安装 <https://www.paddlepaddle.org.cn/install/quick>`__\ 查看。

二、导入飞桨
-----------

安装好飞桨后你就可以在Python程序中导入飞桨。

.. code:: ipython3

    import paddle    
    print(paddle.__version__)

.. parsed-literal::

    2.0.1


三、实践：手写数字识别任务
---------------------------

简单的说，深度学习任务一般分为几个核心步骤：1.数据集的准备和加载；2.模型构建；3.模型训练；4.模型评估。接下来你可以使用飞桨框架API，一步步实现上述步骤。

3.1 加载内置数据集
~~~~~~~~~~~~~~~~~

飞桨框架内置了一些常见的数据集，在这个示例中，你可以加载飞桨框架的内置数据集：手写数字体数据集。这里加载两个数据集，一个用来训练模型，一个用来评估模型。

.. code:: ipython3
    
    from paddle.vision.transforms import ToTensor
    
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
    val_dataset =  paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

3.2 模型搭建
~~~~~~~~~~~~

通过 ``Sequential`` 将一层一层的网络结构组建起来。注意，需要先对数据进行 ``Flatten`` 操作，将[1, 28, 28]形状的图片数据改变形状为[1, 784]。

.. code:: ipython3

    mnist = paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(784, 512),
        paddle.nn.ReLU(),
        paddle.nn.Dropout(0.2),
        paddle.nn.Linear(512, 10)
    )

3.3 模型训练
~~~~~~~~~~~~

在训练模型前，需要配置训练模型时损失的计算方法与优化方法，你可以使用飞桨框架提供的 ``prepare`` 完成，之后使用 ``fit`` 接口来开始训练模型。

.. code:: ipython3
    
    # 预计模型结构生成模型对象，便于进行后续的配置、训练和验证
    model = paddle.Model(mnist)  
    
    # 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
    model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
                  paddle.nn.CrossEntropyLoss(),
                  paddle.metric.Accuracy())
    
    # 开始模型训练
    model.fit(train_dataset,
              epochs=5, 
              batch_size=64,
              verbose=1)


.. parsed-literal::

    The loss value printed in the log is the current step, and the metric is the average value of previous step.
    Epoch 1/5
    step 938/938 [==============================] - loss: 0.1161 - acc: 0.9298 - 16ms/step          
    Epoch 2/5
    step 938/938 [==============================] - loss: 0.0272 - acc: 0.9691 - 15ms/step          
    Epoch 3/5
    step 938/938 [==============================] - loss: 0.0326 - acc: 0.9789 - 16ms/step          
    Epoch 4/5
    step 938/938 [==============================] - loss: 0.0043 - acc: 0.9826 - 16ms/step          
    Epoch 5/5
    step 938/938 [==============================] - loss: 0.0853 - acc: 0.9863 - 15ms/step          

3.4 模型评估
~~~~~~~~~~~~

你可以使用预先定义的验证数据集来评估前一步训练得到的模型的精度。

.. code:: ipython3

    model.evaluate(val_dataset, verbose=0)


.. parsed-literal::

    {'loss': [1.0728842e-06], 'acc': 0.9822}


可以看出，初步训练得到的模型效果在98%附近，在逐渐了解飞桨后，你可以通过调整其中的训练参数来提升模型的精度。

至此你就通过飞桨几个简单的API完成了一个深度学习任务，你也可以针对自己的需求来更换其中的代码，比如对数据集进行增强、使用 ``CNN`` 模型等，飞桨官网提供了丰富的教程与案例可供参考。
