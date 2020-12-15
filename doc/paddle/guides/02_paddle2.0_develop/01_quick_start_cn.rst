.. _cn_doc_quick_start:

10分钟快速上手飞桨（PaddlePaddle）
=================================

本示例通过一个基础案例带您从一个飞桨新手快速掌握如何使用。

1. 安装飞桨
-----------

如果您已经安装好飞桨那么可以跳过此步骤。我们针对用户提供了一个方便易用的安装引导页面，您可以通过选择自己的系统和软件版本来获取对应的安装命令，具体可以点击\ `快速安装 <https://www.paddlepaddle.org.cn/install/quick>`__\ 查看。

2. 导入飞桨
-----------

安装好飞桨后我们就可以在Python程序中进行飞桨的导入。

.. code:: ipython3

    import paddle    
    print(paddle.__version__)

.. parsed-literal::

    2.0.0-rc1


3. 实践一个手写数字识别任务
---------------------------

对于深度学习任务如果简单来看，其实分为几个核心步骤：1.
数据集的准备和加载；2.
模型的构建；3.模型训练；4.模型评估。那么接下来我们就一步一步带您通过飞桨的少量API快速实现。

3.1 数据加载
~~~~~~~~~~~~

加载我们框架为您准备好的一个手写数字识别数据集。这里我们使用两个数据集，一个用来做模型的训练，一个用来做模型的评估。

.. code:: ipython3
    
    from paddle.vision.transforms import ToTensor
    
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
    val_dataset =  paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())

3.2 模型搭建
~~~~~~~~~~~~

通过Sequential将一层一层的网络结构组建起来。需要先对数据进行Flatten操作，将[1,28, 28]形状的图片数据改变形状为[1,784]。

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

配置好我们模型训练需要的损失计算方法和优化方法后就可以使用fit接口来开启我们的模型训练过程。

.. code:: ipython3
    
    # 预计模型结构生成模型实例，便于进行后续的配置、训练和验证
    model = paddle.Model(mnist)  
    
    # 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
    model.prepare(paddle.optimizer.Adam(parameters=mnist.parameters()),
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
    step 938/938 [==============================] - loss: 0.1358 - acc: 0.9284 - 18ms/step          
    Epoch 2/5
    step 938/938 [==============================] - loss: 0.0370 - acc: 0.9680 - 18ms/step          
    Epoch 3/5
    step 938/938 [==============================] - loss: 0.0284 - acc: 0.9780 - 18ms/step          
    Epoch 4/5
    step 938/938 [==============================] - loss: 0.0062 - acc: 0.9823 - 18ms/step          
    Epoch 5/5
    step 938/938 [==============================] - loss: 0.0924 - acc: 0.9859 - 18ms/step

3.4 模型评估
~~~~~~~~~~~~

使用我们刚才训练得到的模型参数进行模型的评估操作，看看我们的模型精度如何。

.. code:: ipython3

    model.evaluate(val_dataset, verbose=0)


.. parsed-literal::

    {'loss': [0.0], 'acc': 0.9804}


那么初步训练得到的模型效果在98%附近，我们可以进一步通过调整其中的训练参数来提升我们的模型精度。

至此我们可以知道如何通过飞桨的几个简单API来快速完成一个深度学习任务，大家可以针对自己的需求来更换其中的代码，如果需要使用自己的数据集，那么可以更换数据集加载部分程序，如果需要替换模型，那么可以更改模型代码实现等等。后文会具体描述深度学习每个环节。
