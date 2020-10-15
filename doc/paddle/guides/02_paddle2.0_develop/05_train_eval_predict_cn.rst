.. _cn_doc_train_predict:

训练与预测
=====================

在完成数据预处理，数据加载与模型的组建后，就可以进行模型的训练与预测了。飞桨框架提供了两种训练与预测的方法，一种是用 `paddle.Model` 对模型进行封装，通过高层API如 `Model.fit()、Model.evaluate()、Model.predict()` 等完成模型的训练与预测；另一种就是基于基础API常规的训练方式。

.. note::

    高层API实现的模型训练与预测如 `Model.fit()、Model.evaluate()、Model.predict()` 都可以通过基础API实现，本文先介绍高层API的训练方式，然后会将高层API拆解为基础API的方式，方便读者对比学习。


在封装模型前，需要先完成数据的加载与模型的组建，由于这一部分高层API与基础API通用，所以都可用下面的代码：

.. code:: ipython3

    import paddle

    # 加载数据集
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    test_dataset = paddle.vision.datasets.MNIST(mode='test')
    
    # 定义网络结构
    mnist = paddle.nn.Sequential(
        paddle.nn.Flatten(1, -1),
        paddle.nn.Linear(784, 512),
        paddle.nn.ReLU(),
        paddle.nn.Dropout(0.2),
        paddle.nn.Linear(512, 10)
    )

通过上述的代码，我们完成了训练集与测试集的构建，并创建了一个 mnist的网络模型。下面就让我们分别用两种方式完成模型的训练与预测。

1. 通过 `paddle.Model` 训练与预测
---------------------------------------------------

我们来使用\ ``paddle.Model``\ 完成模型的封装，将网络结构组合成一个可快速使用高层API进行训练和预测的类。代码如下：

.. code:: ipython3

    model = paddle.Model(mnist)

1.1 用 `Model.prepare()` 配置模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用 `paddle.Model` 完成模型的封装后，在训练前，需要对模型进行配置，通过\ ``Model.prepare``\ 接口来对训练进行提前的配置准备工作，包括设置模型优化器，Loss计算方法，精度计算方法等。

.. code:: ipython3

    # 为模型训练做准备，设置优化器，损失函数和精度计算方式
    model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters()), 
                  loss=paddle.nn.CrossEntropyLoss(),
                  metrics=paddle.metric.Accuracy())

1.2 用 `Model.fit()` 训练模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

做好模型训练的前期准备工作后，我们正式调用\ ``fit()``\ 接口来启动训练过程，需要指定一下至少3个关键参数：训练数据集，训练轮次和单次训练数据批次大小。

.. code:: ipython3

    # 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
    model.fit(train_dataset, 
              epochs=10, 
              batch_size=32,
              verbose=1)


.. parsed-literal::

    Epoch 1/5
    step 1875/1875 [==============================] - loss: 0.1171 - acc: 0.8737 - 7ms/step              
    Epoch 2/5
    step 1875/1875 [==============================] - loss: 0.2732 - acc: 0.9082 - 10ms/step           
    Epoch 3/5
    step 1875/1875 [==============================] - loss: 0.1402 - acc: 0.9111 - 10ms/step           
    Epoch 4/5
    step 1875/1875 [==============================] - loss: 0.0791 - acc: 0.9170 - 12ms/step              
    Epoch 5/5
    step 1875/1875 [==============================] - loss: 0.3515 - acc: 0.9231 - 11ms/step           


1.3 用 `Model.evaluate()` 评估模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于训练好的模型进行评估操作可以使用\ ``evaluate``\ 接口来实现，事先定义好用于评估使用的数据集后，可以简单的调用\ ``evaluate``\ 接口即可完成模型评估操作，结束后根据prepare中loss和metric的定义来进行相关评估结果计算返回。

返回格式是一个字典： \* 只包含loss，\ ``{'loss': xxx}`` \*
包含loss和一个评估指标，\ ``{'loss': xxx, 'metric name': xxx}`` \*
包含loss和多个评估指标，\ ``{'loss': xxx, 'metric name': xxx, 'metric name': xxx}``

.. code:: ipython3

    # 用 evaluate 在测试集上对模型进行验证
    eval_result = model.evaluate(test_dataset, verbose=1)


.. parsed-literal::

    Eval begin...
    step 10000/10000 [==============================] - loss: 0.0000e+00 - acc: 0.9427 - 2ms/step
    Eval samples: 10000


1.4 用 `Model.predict()` 预测模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
高层API中提供了\ ``predict``\ 接口来方便用户对训练好的模型进行预测验证，只需要基于训练好的模型将需要进行预测测试的数据放到接口中进行计算即可，接口会将经过模型计算得到的预测结果进行返回。

返回格式是一个list，元素数目对应模型的输出数目： \*
模型是单一输出：[(numpy_ndarray_1, numpy_ndarray_2, …, numpy_ndarray_n)]
\* 模型是多输出：[(numpy_ndarray_1, numpy_ndarray_2, …,
numpy_ndarray_n), (numpy_ndarray_1, numpy_ndarray_2, …,
numpy_ndarray_n), …]

numpy_ndarray_n是对应原始数据经过模型计算后得到的预测数据，数目对应预测数据集的数目。

.. code:: ipython3

    # 用 predict 在测试集上对模型进行测试
    test_result = model.predict(test_dataset)

.. parsed-literal::

    Predict begin...
    step 10000/10000 [==============================] - 1ms/step           
    Predict samples: 10000


2. 通过基础API实现模型的训练与预测
-----------------------------------------

除了通过第一部分的高层API实现模型的训练与预测，飞桨框架也同样支持通过基础API对模型进行训练与预测。简单来说， `Model.prepare()、Model.fit()、Model.evaluate()、Model.predict()` 都是由基础API封装而来。下面通过拆解高层API到基础API的方式，来了解一下如何用基础API完成模型的训练与预测。

2.1 拆解 `Model.prepare()、Model.fit()` -- 用基础API训练模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

飞桨框架通过基础API对模型进行训练与预测，对应第一部分的 `Model.prepare()` 与 `Model.fit()` ：

.. code:: ipython3
    
    # dataset与mnist的定义与第一部分内容一致

    # 用 DataLoader 实现数据加载
    train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=32, shuffle=True)
    
    mnist.train()
    
    # 设置迭代次数
    epochs = 5
    
    # 设置优化器
    optim = paddle.optimizer.Adam(parameters=model.parameters())
    
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            
            x_data = data[0]            # 训练数据
            y_data = data[1]            # 训练数据标签
            predicts = mnist(x_data)    # 预测结果  
            
            # 计算损失 等价于 prepare 中loss的设置
            loss = paddle.nn.functional.cross_entropy(predicts, y_data)
            
            # 计算准确率 等价于 prepare 中metrics的设置
            acc = paddle.metric.accuracy(predicts, y_data)
            
            # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中

            # 反向传播 
            loss.backward()
            
            if (batch_id+1) % 1800 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))

            # 更新参数 
            optim.step()

            # 梯度清零
            optim.clear_grad()


.. parsed-literal::

    epoch: 0, batch_id: 1799, loss is: [0.1566943], acc is: [0.96875]
    epoch: 1, batch_id: 1799, loss is: [0.9046197], acc is: [0.9375]
    epoch: 2, batch_id: 1799, loss is: [0.5993278], acc is: [0.96875]
    epoch: 3, batch_id: 1799, loss is: [0.13339292], acc is: [0.96875]
    epoch: 4, batch_id: 1799, loss is: [0.8168528], acc is: [0.9375]


2.2 拆解 `Model.evaluate()` -- 用基础API验证模型

飞桨框架通过基础API对模型进行验证，对应第一部分的 `Model.evaluate()`:

.. code:: ipython3

    # 加载测试数据集
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=32, drop_last=True)
    
    mnist.eval()

    for batch_id, data in enumerate(test_loader()):
        
        x_data = data[0]            # 测试数据
        y_data = data[1]            # 测试数据标签
        predicts = mnist(x_data)    # 预测结果
        
        # 计算损失与精度
        loss = paddle.nn.functional.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        
        # 打印信息
        if (batch_id+1) % 300 == 0:
            print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))

.. parsed-literal::

    batch_id: 299, loss is: [0.12025189], acc is: [0.9375]


2.3 拆解 `Model.predict()` -- 用基础API测试模型

飞桨框架通过基础API对模型进行测试，对应第一部分的 `Model.predict()`:

.. code:: ipython3

    # 加载测试数据集
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=32, drop_last=True)

    mnist.eval()
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0] 
        predicts = mnist(x_data)
        # 获取预测结果
    print("predict finished")


.. parsed-literal::

    predict finished
