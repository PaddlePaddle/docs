.. _cn_doc_train_eval_predict:

训练与预测
=====================

在完成数据预处理，数据加载与模型的组建后，你就可以进行模型的训练与预测了。飞桨主框架提供了两种训练与预测的方法，一种是用\ ``paddle.Model``\ 对模型进行封装，通过高层API如\ ``Model.fit()、Model.evaluate()、Model.predict()``\ 等完成模型的训练与预测；另一种就是基于基础API常规的训练方式。

.. note::

    高层API实现的模型训练与预测如\ ``Model.fit()、Model.evaluate()、Model.predict()``\ 都可以通过基础API实现，本文先介绍高层API的训练方式，然后会将高层API拆解为基础API的方式，方便对比学习。最后会补充介绍如何使用paddle inference进行预测。

一、训练前准备
---------------------

在封装模型前，需要先完成数据的加载，由于这一部分高层API与基础API通用，所以都可用下面的代码实现：

.. code:: ipython3

    import paddle
    from paddle.vision.transforms import ToTensor

    # 加载数据集
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())
    test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())


通过上述的代码，你就完成了训练集与测试集的构建，下面分别用两种方式完成模型的训练与预测。

二、通过\ ``paddle.Model``\ 训练与预测
------------------------------------

在这里你可以采用Sequential组网或者SubClass 组网的方式来创建一个mnist网络模型，你可使用\ ``paddle.Model``\ 完成模型的封装，将网络结构组合成一个可快速使用高层API进行训练和预测的对象。代码如下：

.. code:: ipython3

    # 定义网络结构(采用 Sequential组网方式 )
    mnist = paddle.nn.Sequential(
        paddle.nn.Flatten(1, -1),
        paddle.nn.Linear(784, 512),
        paddle.nn.ReLU(),
        paddle.nn.Dropout(0.2),
        paddle.nn.Linear(512, 10)
    )


    model = paddle.Model(mnist)

2.1 用\ ``Model.prepare()``\ 配置模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用\ ``paddle.Model``\ 完成模型的封装后，在训练前，需要对模型进行配置，通过\ ``Model.prepare``\ 接口来对训练进行提前的配置准备工作，包括设置模型优化器，Loss计算方法，精度计算方法等。

.. code:: ipython3

    # 为模型训练做准备，设置优化器，损失函数和精度计算方式
    model.prepare(optimizer=paddle.optimizer.Adam(parameters=model.parameters()), 
                  loss=paddle.nn.CrossEntropyLoss(),
                  metrics=paddle.metric.Accuracy())

2.2 用\ ``Model.fit()``\ 训练模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

做好模型训练的前期准备工作后，调用\ ``fit()``\ 接口来启动训练过程，需要指定至少3个关键参数：训练数据集，训练轮次和单次训练数据批次大小。

.. code:: ipython3

    # 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
    model.fit(train_dataset, 
              epochs=5, 
              batch_size=64,
              verbose=1)


.. parsed-literal::

    The loss value printed in the log is the current step, and the metric is the average value of previous step.
    Epoch 1/5
    step 938/938 [==============================] - loss: 0.1785 - acc: 0.9281 - 19ms/step          
    Epoch 2/5
    step 938/938 [==============================] - loss: 0.0365 - acc: 0.9688 - 19ms/step          
    Epoch 3/5
    step 938/938 [==============================] - loss: 0.0757 - acc: 0.9781 - 19ms/step          
    Epoch 4/5
    step 938/938 [==============================] - loss: 0.0054 - acc: 0.9824 - 19ms/step          
    Epoch 5/5
    step 938/938 [==============================] - loss: 0.0640 - acc: 0.9858 - 19ms/step  

2.3 用\ ``Model.evaluate()``\ 评估模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于训练好的模型进行评估可以使用\ ``evaluate``\ 接口，事先定义好用于评估使用的数据集后，直接调用\ ``evaluate``\ 接口即可完成模型评估操作，结束后根据在\ ``prepare``\ 中\ ``loss``\ 和\ ``metric``\ 的定义来进行相关评估结果计算返回。

返回格式是一个字典： \* 只包含loss，\ ``{'loss': xxx}`` \*
包含loss和一个评估指标，\ ``{'loss': xxx, 'metric name': xxx}`` \*
包含loss和多个评估指标，\ ``{'loss': xxx, 'metric name1': xxx, 'metric name2': xxx}``

.. code:: ipython3

    # 用 evaluate 在测试集上对模型进行验证
    eval_result = model.evaluate(test_dataset, verbose=1)


.. parsed-literal::

    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 10000/10000 [==============================] - loss: 3.5763e-07 - acc: 0.9809 - 2ms/step
    Eval samples: 10000

2.4 用\ ``Model.predict()``\ 预测模型
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
    step 10000/10000 [==============================] - 2ms/step           
    Predict samples: 10000


三、通过基础API实现模型的训练与预测
-----------------------------------------

除了通过第一部分的高层API实现模型的训练与预测，飞桨框架也同样支持通过基础API对模型进行训练与预测。简单来说，\ ``Model.prepare()、Model.fit()、Model.evaluate()、Model.predict()``\ 都是由基础API封装而来。下面通过拆解高层API到基础API的方式，来了解如何用基础API完成模型的训练与预测。


.. note::

    对于网络模型的创建你依旧可以选择Sequential组网方式，也可以采用SubClass组网方式，为方便后续使用paddle inference进行预测，我们使用SubClass组网方式创建网络，若后续使用paddle inference预测，需通过paddle.jit.save保存适用于预测部署的模型，并在forward函数前加@paddle.jit.to_static装饰器，将函数内的动态图API转化为静态图API。

.. code:: ipython3

    # 定义网络结构( 采用SubClass 组网 )
    class Mnist(paddle.nn.Layer):
        def __init__(self):
            super(Mnist, self).__init__()
            self.flatten = paddle.nn.Flatten()
            self.linear_1 = paddle.nn.Linear(784, 512)
            self.linear_2 = paddle.nn.Linear(512, 10)
            self.relu = paddle.nn.ReLU()
            self.dropout = paddle.nn.Dropout(0.2)
       
        #后续若不使用paddle inferece，可对 @paddle.jit.to_static 进行注释  
        @paddle.jit.to_static       
        def forward(self, inputs):
            y = self.flatten(inputs)
            y = self.linear_1(y)
            y = self.relu(y)
            y = self.dropout(y)
            y = self.linear_2(y)
            return y


3.1 拆解\ ``Model.prepare()、Model.fit()``\ -- 用基础API训练模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

飞桨框架通过基础API对模型进行训练与预测，对应第一部分的\ ``Model.prepare()``\ 与\ ``Model.fit()``\ ：

.. code:: ipython3

    # dataset与mnist的定义与第一部分内容一致

    # 用 DataLoader 实现数据加载
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    mnist=Mnist()
    mnist.train()
    
    # 设置迭代次数
    epochs = 5
    
    # 设置优化器
    optim = paddle.optimizer.Adam(parameters=mnist.parameters())
    # 设置损失函数
    loss_fn = paddle.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            
            x_data = data[0]            # 训练数据
            y_data = data[1]            # 训练数据标签
            predicts = mnist(x_data)    # 预测结果  
            
            # 计算损失 等价于 prepare 中loss的设置
            loss = loss_fn(predicts, y_data)
            
            # 计算准确率 等价于 prepare 中metrics的设置
            acc = paddle.metric.accuracy(predicts, y_data)
            
            # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中

            # 反向传播 
            loss.backward()
            
            if (batch_id+1) % 900 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id+1, loss.numpy(), acc.numpy()))

            # 更新参数 
            optim.step()

            # 梯度清零
            optim.clear_grad()
    ##保存模型，会生成*.pdmodel、*.pdiparams、*.pdiparams.info三个模型文件
    path='./mnist/inference_model'
    paddle.jit.save(layer=mnist,path=path)


.. parsed-literal::

    epoch: 0, batch_id: 900, loss is: [0.29550618], acc is: [0.90625]
    epoch: 1, batch_id: 900, loss is: [0.05875912], acc is: [0.984375]
    epoch: 2, batch_id: 900, loss is: [0.05824642], acc is: [0.96875]
    epoch: 3, batch_id: 900, loss is: [0.02940615], acc is: [1.]
    epoch: 4, batch_id: 900, loss is: [0.05713747], acc is: [0.984375]

3.2 拆解\ ``Model.evaluate()``\ -- 用基础API验证模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

飞桨框架通过基础API对模型进行验证，对应第一部分的\ ``Model.evaluate()``\ :

.. code:: ipython3

    # 加载测试数据集
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, drop_last=True)
    loss_fn = paddle.nn.CrossEntropyLoss()

    mnist.eval()

    for batch_id, data in enumerate(test_loader()):
        
        x_data = data[0]            # 测试数据
        y_data = data[1]            # 测试数据标签
        predicts = mnist(x_data)    # 预测结果
        
        # 计算损失与精度
        loss = loss_fn(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        
        # 打印信息
        if (batch_id+1) % 30 == 0:
            print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id+1, loss.numpy(), acc.numpy()))

.. parsed-literal::

    batch_id: 30, loss is: [0.15860887], acc is: [0.953125]
    batch_id: 60, loss is: [0.21005578], acc is: [0.921875]
    batch_id: 90, loss is: [0.0889321], acc is: [0.953125]
    batch_id: 120, loss is: [0.00115552], acc is: [1.]
    batch_id: 150, loss is: [0.12016675], acc is: [0.984375]


3.3 拆解\ ``Model.predict()``\ -- 用基础API测试模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

飞桨框架通过基础API对模型进行测试，对应第一部分的\ ``Model.predict()``\ :

.. code:: ipython3

    # 加载测试数据集
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, drop_last=True)

    mnist.eval()
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0] 
        predicts = mnist(x_data)
        # 获取预测结果
    print("predict finished")


.. parsed-literal::

    predict finished
    

部署预测模型
=====================
其中预测方法除以上两种外，还可采用原生推理库paddle inference 进行推理部署，该方法支持TeansorRT加速，支持第三方框架模型，支持量化、裁剪后的模型，适合于工业部署或对推理性能、通用性有要求的用户。

 
四、通过paddle inference实现预测
-----------------------------------------

paddle inference与model.predict()以及基础API的预测相比，可使用MKLDNN、CUDNN、TensorRT进行预测加速，同时支持用 X2Paddle 工具从第三方框架（TensorFlow、Pytorh 、 Caffe 等）产出的模型，可联动PaddleSlim，支持加载量化、裁剪和蒸馏后的模型部署。针对不同平台不同的应用场景进行了深度的适配优化，保证模型在服务器端即训即用，快速部署。在这里，我们只简单的展示如何用paddle inference实现该模型的部署预测。

4.1 准备预测部署模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
要使用paddle inference预测需得到paddle预测格式的模型，所以你需要在训练过程中通过 paddle.jit.save(layer=mnist,path=path) 来保存模型，注意在训练时在forward函数前加@paddle.jit.to_static装饰器，将函数内的动态图API转化为静态图API。在第三章节基础API模型的训练中已加入相关配置。

.. code:: ipython3

    #模型目录如下：
                mnist/
            ├── inference.pdmodel
            ├── inference.pdiparams.info
            └── inference.pdiparams
4.2 准备预测部署程序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
将以下代码保存为python_demo.py文件：

.. code:: ipython3

    import argparse
    import numpy as np
    from skimage import transform,data

    # 引用 paddle inference 预测库
    import paddle.inference as paddle_infer
    from PIL import Image

    def main():
        args = parse_args()

        # 创建 config
        config = paddle_infer.Config(args.model_file, args.params_file)

        # 根据 config 创建 predictor
        predictor = paddle_infer.create_predictor(config)

        # 获取输入的名称
        input_names = predictor.get_input_names()
        input_handle = predictor.get_input_handle(input_names[0])

        # 设置输入，自定义一张输入照片，图片大小为28*28
        im=Image.open('./img3.png').convert('L')
        im=np.array(im).reshape(1,1,28,28).astype(np.float32)
        input_handle.copy_from_cpu(im)

        # 运行predictor
        predictor.run()

        # 获取输出
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu() # numpy.ndarray类型，是10个分类的概率
        print(output_data)
        print("Output data size is {}".format(output_data.size))
        print("Output data shape is {}".format(output_data.shape))
        pred=np.argmax(output_data) #选出概率最大的一个
        print("The predicted data is ： {}".format(pred.item()))

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_file", type=str, help="model filename")
        parser.add_argument("--params_file", type=str, help="parameter filename")
        parser.add_argument("--batch_size", type=int, default=1, help="batch size")
        return parser.parse_args()

    if __name__ == "__main__":
        main()
        

4.3 执行预测程序
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

   python python_demo.py --model_file ./mnist/inference_model.pdmodel --params_file ./mnist/inference_model.pdiparams --batch_size 2

.. parsed-literal::
    
    #输出如下
    
    [[-1347.5923  -1156.918    -774.73865  3387.0623  -1553.3696    107.96879
      -2631.2185   -701.50323 -1094.3896    206.71666]]
    Output data size is 10
    Output data shape is (1, 10)
    The predicted data is ： 3
    
详细教程可参照paddle inference文档：https://paddle-inference.readthedocs.io/en/latest/quick_start/python_demo.html

