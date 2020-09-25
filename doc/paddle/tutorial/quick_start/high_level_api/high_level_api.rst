飞桨高层API使用指南
===================

1. 简介
-------

飞桨框架2.0全新推出高层API，是对飞桨API的进一步封装与升级，提供了更加简洁易用的API，进一步提升了飞桨的易学易用性，并增强飞桨的功能。

飞桨高层API面向从深度学习小白到资深开发者的所有人群，对于AI初学者来说，使用高层API可以简单快速的构建深度学习项目，对于资深开发者来说，可以快速完成算法迭代。

飞桨高层API具有以下特点：

-  易学易用:
   高层API是对普通动态图API的进一步封装和优化，同时保持与普通API的兼容性，高层API使用更加易学易用，同样的实现使用高层API可以节省大量的代码。
-  低代码开发:
   使用飞桨高层API的一个明显特点是，用户可编程代码量大大缩减。
-  动静转换:
   高层API支持动静转换，用户只需要改一行代码即可实现将动态图代码在静态图模式下训练，既方便用户使用动态图调试模型，又提升了模型训练效率。

在功能增强与使用方式上，高层API有以下升级：

-  模型训练方式升级:
   高层API中封装了Model类，继承了Model类的神经网络可以仅用几行代码完成模型的训练。
-  新增图像处理模块transform:
   飞桨新增了图像预处理模块，其中包含数十种数据处理函数，基本涵盖了常用的数据处理、数据增强方法。
-  提供常用的神经网络模型可供调用:
   高层API中集成了计算机视觉领域和自然语言处理领域常用模型，包括但不限于mobilenet、resnet、yolov3、cyclegan、bert、transformer、seq2seq等等。同时发布了对应模型的预训练模型，用户可以直接使用这些模型或者在此基础上完成二次开发。

2. 安装并使用飞桨高层API
------------------------

飞桨高层API无需独立安装，只需要安装好paddlepaddle即可，安装完成后import
paddle即可使用相关高层API，如：paddle.Model、视觉领域paddle.vision、NLP领域paddle.text。

.. code:: ipython3

    import paddle
    import paddle.vision as vision
    import paddle.text as text
    
    # 启动动态图训练模式
    paddle.disable_static()
    
    paddle.__version__




.. parsed-literal::

    '2.0.0-beta0'



2. 目录
-------

本指南教学内容覆盖

-  使用高层API提供的自带数据集进行相关深度学习任务训练。
-  使用自定义数据进行数据集的定义、数据预处理和训练。
-  如何在数据集定义和加载中应用数据增强相关接口。
-  如何进行模型的组网。
-  高层API进行模型训练的相关API使用。
-  如何在fit接口满足需求的时候进行自定义，使用基础API来完成训练。
-  如何使用多卡来加速训练。

3. 数据集定义、加载和数据预处理
-------------------------------

对于深度学习任务，均是框架针对各种类型数字的计算，是无法直接使用原始图片和文本等文件来完成。那么就是涉及到了一项动作，就是将原始的各种数据文件进行处理加工，转换成深度学习任务可以使用的数据。

3.1 框架自带数据集使用
~~~~~~~~~~~~~~~~~~~~~~

高层API将一些我们常用到的数据集作为领域API对用户进行开放，对应API所在目录为\ ``paddle.vision.datasets``\ ，那么我们先看下提供了哪些数据集。

.. code:: ipython3

    print('视觉相关数据集：', paddle.vision.datasets.__all__)
    print('自然语言相关数据集：', paddle.text.datasets.__all__)


.. parsed-literal::

    视觉相关数据集： ['DatasetFolder', 'ImageFolder', 'MNIST', 'Flowers', 'Cifar10', 'Cifar100', 'VOC2012']
    自然语言相关数据集： ['Conll05st', 'Imdb', 'Imikolov', 'Movielens', 'UCIHousing', 'WMT14', 'WMT16']


这里我们是加载一个手写数字识别的数据集，用\ ``mode``\ 来标识是训练数据还是测试数据集。数据集接口会自动从远端下载数据集到本机缓存目录\ ``~/.cache/paddle/dataset``\ 。

.. code:: ipython3

    # 训练数据集
    train_dataset = vision.datasets.MNIST(mode='train')
    
    # 验证数据集
    val_dataset = vision.datasets.MNIST(mode='test')

3.2 自定义数据集
~~~~~~~~~~~~~~~~

更多的时候我们是需要自己使用已有的相关数据来定义数据集，那么这里我们通过一个案例来了解如何进行数据集的定义，飞桨为用户提供了\ ``paddle.io.Dataset``\ 基类，让用户通过类的集成来快速实现数据集定义。

.. code:: ipython3

    from paddle.io import Dataset
    
    
    class MyDataset(Dataset):
        """
        步骤一：继承paddle.io.Dataset类
        """
        def __init__(self, mode='train'):
            """
            步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
            """
            super(MyDataset, self).__init__()
    
            if mode == 'train':
                self.data = [
                    ['traindata1', 'label1'],
                    ['traindata2', 'label2'],
                    ['traindata3', 'label3'],
                    ['traindata4', 'label4'],
                ]
            else:
                self.data = [
                    ['testdata1', 'label1'],
                    ['testdata2', 'label2'],
                    ['testdata3', 'label3'],
                    ['testdata4', 'label4'],
                ]
        
        def __getitem__(self, index):
            """
            步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
            """
            data = self.data[index][0]
            label = self.data[index][1]
    
            return data, label
    
        def __len__(self):
            """
            步骤四：实现__len__方法，返回数据集总数目
            """
            return len(self.data)
    
    # 测试定义的数据集
    train_dataset_2 = MyDataset(mode='train')
    val_dataset_2 = MyDataset(mode='test')
    
    print('=============train dataset=============')
    for data, label in train_dataset:
        print(data, label)
    
    print('=============evaluation dataset=============')
    for data, label in val_dataset:
        print(data, label)


.. parsed-literal::

    =============train dataset=============
    traindata1 label1
    traindata2 label2
    traindata3 label3
    traindata4 label4
    =============evaluation dataset=============
    testdata1 label1
    testdata2 label2
    testdata3 label3
    testdata4 label4


3.3 数据增强
~~~~~~~~~~~~

训练过程中有时会遇到过拟合的问题，其中一个解决方法就是对训练数据做增强，对数据进行处理得到不同的图像，从而泛化数据集。数据增强API是定义在领域目录的transofrms下，这里我们介绍两种使用方式，一种是基于框架自带数据集，一种是基于自己定义的数据集。

3.3.1 框架自带数据集
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    from paddle.vision.transforms import Compose, Resize, ColorJitter
    
    
    # 定义想要使用那些数据增强方式，这里用到了随机调整亮度、对比度和饱和度，改变图片大小
    transform = Compose([ColorJitter(), Resize(size=100)])
    
    # 通过transform参数传递定义好的数据增项方法即可完成对自带数据集的应用
    train_dataset_3 = vision.datasets.MNIST(mode='train', transform=transform)

3.3.2 自定义数据集
^^^^^^^^^^^^^^^^^^

针对自定义数据集使用数据增强有两种方式，一种是在数据集的构造函数中进行数据增强方法的定义，之后对__getitem__中返回的数据进行应用。另外一种方式也可以给自定义的数据集类暴漏一个构造参数，在实例化类的时候将数据增强方法传递进去。

.. code:: ipython3

    from paddle.io import Dataset
    
    
    class MyDataset(Dataset):
        def __init__(self, mode='train'):
            super(MyDataset, self).__init__()
    
            if mode == 'train':
                self.data = [
                    ['traindata1', 'label1'],
                    ['traindata2', 'label2'],
                    ['traindata3', 'label3'],
                    ['traindata4', 'label4'],
                ]
            else:
                self.data = [
                    ['testdata1', 'label1'],
                    ['testdata2', 'label2'],
                    ['testdata3', 'label3'],
                    ['testdata4', 'label4'],
                ]
    
            # 定义要使用的数据预处理方法，针对图片的操作
            self.transform = Compose([ColorJitter(), Resize(size=100)])
        
        def __getitem__(self, index):
            data = self.data[index][0]
    
            # 在这里对训练数据进行应用
            # 这里只是一个示例，测试时需要将数据集更换为图片数据进行测试
            data = self.transform(data)
    
            label = self.data[index][1]
    
            return data, label
    
        def __len__(self):
            return len(self.data)

4. 模型组网
-----------

针对高层API在模型组网上和基础API是统一的一套，无需投入额外的学习使用成本。那么这里我举几个简单的例子来做示例。

4.1 Sequential组网
~~~~~~~~~~~~~~~~~~

针对顺序的线性网络结构我们可以直接使用Sequential来快速完成组网，可以减少类的定义等代码编写。

.. code:: ipython3

    # Sequential形式组网
    mnist = paddle.nn.Sequential(
        paddle.nn.Flatten(),
        paddle.nn.Linear(784, 512),
        paddle.nn.ReLU(),
        paddle.nn.Dropout(0.2),
        paddle.nn.Linear(512, 10)
    )

4.2 SubClass组网
~~~~~~~~~~~~~~~~

针对一些比较复杂的网络结构，就可以使用Layer子类定义的方式来进行模型代码编写，在\ ``__init__``\ 构造函数中进行组网Layer的声明，在\ ``forward``\ 中使用声明的Layer变量进行前向计算。子类组网方式也可以实现sublayer的复用，针对相同的layer可以在构造函数中一次性定义，在forward中多次调用。

.. code:: ipython3

    # Layer类继承方式组网
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
    
    mnist_2 = Mnist()

4.3 模型封装
~~~~~~~~~~~~

定义好网络结构之后我们来使用\ ``paddle.Model``\ 完成模型的封装，将网络结构组合成一个可快速使用高层API进行训练、评估和预测的类。

在封装的时候我们有两种场景，动态图训练模式和静态图训练模式。

.. code:: ipython3

    # 场景1：动态图模式
    
    # 使用GPU训练
    paddle.set_device('gpu')
    # 模型封装
    model = paddle.Model(mnist)
    
    
    # 场景2：静态图模式
    
    # input = paddle.static.InputSpec([None, 1, 28, 28], dtype='float32')
    # label = paddle.static.InputSpec([None, 1], dtype='int8')
    # model = paddle.Model(mnist, input, label)

4.4 模型可视化
~~~~~~~~~~~~~~

在组建好我们的网络结构后，一般我们会想去对我们的网络结构进行一下可视化，逐层的去对齐一下我们的网络结构参数，看看是否符合我们的预期。这里可以通过\ ``Model.summary``\ 接口进行可视化展示。

.. code:: ipython3

    model.summary((1, 28, 28))


.. parsed-literal::

    --------------------------------------------------------------------------------
       Layer (type)          Input Shape         Output Shape         Param #
    ================================================================================
      Flatten-57509      [-1, 1, 28, 28]            [-1, 784]               0
           Linear-7            [-1, 784]            [-1, 512]         401,920
             ReLU-4            [-1, 512]            [-1, 512]               0
          Dropout-4            [-1, 512]            [-1, 512]               0
           Linear-8            [-1, 512]             [-1, 10]           5,130
    ================================================================================
    Total params: 407,050
    Trainable params: 407,050
    Non-trainable params: 0
    --------------------------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.02
    Params size (MB): 1.55
    Estimated Total Size (MB): 1.57
    --------------------------------------------------------------------------------
    




.. parsed-literal::

    {'total_params': 407050, 'trainable_params': 407050}



另外，summary接口有两种使用方式，下面我们通过两个示例来做展示，除了\ ``Model.summary``\ 这种配套\ ``paddle.Model``\ 封装使用的接口外，还有一套配合没有经过\ ``paddle.Model``\ 封装的方式来使用。可以直接将实例化好的Layer子类放到\ ``paddle.summary``\ 接口中进行可视化呈现。

.. code:: ipython3

    paddle.summary(mnist, (1, 28, 28))


.. parsed-literal::

    --------------------------------------------------------------------------------
       Layer (type)          Input Shape         Output Shape         Param #
    ================================================================================
      Flatten-57508      [-1, 1, 28, 28]            [-1, 784]               0
           Linear-5            [-1, 784]            [-1, 512]         401,920
             ReLU-3            [-1, 512]            [-1, 512]               0
          Dropout-3            [-1, 512]            [-1, 512]               0
           Linear-6            [-1, 512]             [-1, 10]           5,130
    ================================================================================
    Total params: 407,050
    Trainable params: 407,050
    Non-trainable params: 0
    --------------------------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.02
    Params size (MB): 1.55
    Estimated Total Size (MB): 1.57
    --------------------------------------------------------------------------------
    




.. parsed-literal::

    {'total_params': 407050, 'trainable_params': 407050}



这里面有一个注意的点，有的用户可能会疑惑为什么要传递\ ``(1, 28, 28)``\ 这个input_size参数，因为在动态图中，网络定义阶段是还没有得到输入数据的形状信息，我们想要做网络结构的呈现就无从下手，那么我们通过告知接口网络结构的输入数据形状，这样网络可以通过逐层的计算推导得到完整的网络结构信息进行呈现。如果是动态图运行模式，那么就不需要给summary接口传递输入数据形状这个值了，因为在Model封装的时候我们已经定义好了InputSpec，其中包含了输入数据的形状格式。

5. 模型训练
-----------

网络结构通过\ ``paddle.Model``\ 接口封装成模型类后进行执行操作非常的简洁方便，可以直接通过调用\ ``Model.fit``\ 就可以完成训练过程。

使用\ ``Model.fit``\ 接口启动训练前，我们先通过\ ``Model.prepare``\ 接口来对训练进行提前的配置准备工作，包括设置模型优化器，Loss计算方法，精度计算方法等。

.. code:: ipython3

    # 为模型训练做准备，设置优化器，损失函数和精度计算方式
    model.prepare(paddle.optimizer.Adam(parameters=model.parameters()), 
                  paddle.nn.CrossEntropyLoss(),
                  paddle.metric.Accuracy())

做好模型训练的前期准备工作后，我们正式调用\ ``fit()``\ 接口来启动训练过程，需要指定一下至少3个关键参数：训练数据集，训练轮次和单次训练数据批次大小。

.. code:: ipython3

    # 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
    model.fit(train_dataset, 
              epochs=10, 
              batch_size=32,
              verbose=1)


.. parsed-literal::

    Epoch 1/10
    step 1875/1875 [==============================] - loss: 0.1600 - acc: 0.9022 - 10ms/step         
    Epoch 2/10
    step 1875/1875 [==============================] - loss: 0.0455 - acc: 0.9461 - 12ms/step          
    Epoch 3/10
    step 1875/1875 [==============================] - loss: 0.1429 - acc: 0.9544 - 19ms/step          
    Epoch 4/10
    step 1875/1875 [==============================] - loss: 0.0197 - acc: 0.9601 - 22ms/step          
    Epoch 5/10
    step 1875/1875 [==============================] - loss: 0.1762 - acc: 0.9644 - 25ms/step          
    Epoch 6/10
    step 1875/1875 [==============================] - loss: 0.1304 - acc: 0.9667 - 22ms/step          
    Epoch 7/10
    step 1875/1875 [==============================] - loss: 0.0133 - acc: 0.9682 - 22ms/step          
    Epoch 8/10
    step 1875/1875 [==============================] - loss: 0.0097 - acc: 0.9705 - 19ms/step          
    Epoch 9/10
    step 1875/1875 [==============================] - loss: 3.1264e-04 - acc: 0.9716 - 23ms/step      
    Epoch 10/10
    step 1875/1875 [==============================] - loss: 0.0767 - acc: 0.9729 - 13ms/step          


5.1 单机单卡
~~~~~~~~~~~~

我们把刚才单步教学的训练代码做一个整合，这个完整的代码示例就是我们的单机单卡训练程序。

.. code:: ipython3

    
    # 使用GPU训练
    paddle.set_device('gpu')
    
    # 构建模型训练用的Model，告知需要训练哪个模型
    model = paddle.Model(mnist)
    
    # 为模型训练做准备，设置优化器，损失函数和精度计算方式
    model.prepare(paddle.optimizer.Adam(parameters=model.parameters()), 
                  paddle.nn.CrossEntropyLoss(),
                  paddle.metric.Accuracy())
    
    # 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
    model.fit(train_dataset, 
              epochs=10, 
              batch_size=32,
              verbose=1)


.. parsed-literal::

    Epoch 1/10
    step 1875/1875 [==============================] - loss: 0.0490 - acc: 0.9741 - 6ms/step          
    Epoch 2/10
    step 1875/1875 [==============================] - loss: 0.1384 - acc: 0.9760 - 7ms/step          
    Epoch 3/10
    step 1875/1875 [==============================] - loss: 0.0929 - acc: 0.9767 - 7ms/step          
    Epoch 4/10
    step 1875/1875 [==============================] - loss: 0.0190 - acc: 0.9772 - 6ms/step          
    Epoch 5/10
    step 1875/1875 [==============================] - loss: 0.0862 - acc: 0.9774 - 7ms/step          
    Epoch 6/10
    step 1875/1875 [==============================] - loss: 0.0748 - acc: 0.9785 - 8ms/step          
    Epoch 7/10
    step 1875/1875 [==============================] - loss: 0.0039 - acc: 0.9798 - 17ms/step          
    Epoch 8/10
    step 1875/1875 [==============================] - loss: 0.0037 - acc: 0.9808 - 11ms/step          
    Epoch 9/10
    step 1875/1875 [==============================] - loss: 0.0013 - acc: 0.9800 - 8ms/step          
    Epoch 10/10
    step 1875/1875 [==============================] - loss: 0.0376 - acc: 0.9810 - 8ms/step            


5.2 单机多卡
~~~~~~~~~~~~

对于高层API来实现单机多卡非常简单，整个训练代码和单机单卡没有差异。直接使用\ ``paddle.distributed.launch``\ 启动单机单卡的程序即可。

.. code:: bash

   $ python -m paddle.distributed.launch train.py

train.py里面包含的就是单机单卡代码

5.3 自定义Loss
~~~~~~~~~~~~~~

有时我们会遇到特定任务的Loss计算方式在框架既有的Loss接口中不存在，或算法不符合自己的需求，那么期望能够自己来进行Loss的自定义，我们这里就会讲解介绍一下如何进行Loss的自定义操作，首先来看下面的代码：

.. code:: python

   class SelfDefineLoss(paddle.nn.Layer):
       """
       1. 继承paddle.nn.Layer
       """
       def __init__(self):
           """
           2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
           """
           super(SelfDefineLoss, self).__init__()

       def forward(self, input, label):
           """
           3. 实现forward函数，forward在调用时会传递两个参数：input和label
               - input：单个或批次训练数据经过模型前向计算输出结果
               - label：单个或批次训练数据对应的标签数据

               接口返回值是一个Tensor，根据自定义的逻辑加和或计算均值后的损失
           """
           # 使用Paddle中相关API自定义的计算逻辑
           # output = xxxxx
           # return output

那么了解完代码层面如果编写自定义代码后我们看一个实际的例子，下面是在图像分割示例代码中写的一个自定义Loss，当时主要是想使用自定义的softmax计算维度。

.. code:: python

   class SoftmaxWithCrossEntropy(paddle.nn.Layer):
       def __init__(self):
           super(SoftmaxWithCrossEntropy, self).__init__()

       def forward(self, input, label):
           loss = F.softmax_with_cross_entropy(input, 
                                               label, 
                                               return_softmax=False,
                                               axis=1)
           return paddle.mean(loss)

5.4 自定义Metric
~~~~~~~~~~~~~~~~

和Loss一样，如果遇到一些想要做个性化实现的操作时，我们也可以来通过框架完成自定义的评估计算方法，具体的实现方式如下：

.. code:: python

   class SelfDefineMetric(paddle.metric.Metric):
       """
       1. 继承paddle.metric.Metric
       """
       def __init__(self):
           """
           2. 构造函数实现，自定义参数即可
           """
           super(SelfDefineMetric, self).__init__()

       def name(self):
           """
           3. 实现name方法，返回定义的评估指标名字
           """
           return '自定义评价指标的名字'

       def compute(self, ...)
           """
           4. 本步骤可以省略，实现compute方法，这个方法主要用于`update`的加速，可以在这个方法中调用一些paddle实现好的Tensor计算API，编译到模型网络中一起使用低层C++ OP计算。
           """

           return 自己想要返回的数据，会做为update的参数传入。

       def update(self, ...):
           """
           5. 实现update方法，用于单个batch训练时进行评估指标计算。
           - 当`compute`类函数未实现时，会将模型的计算输出和标签数据的展平作为`update`的参数传入。
           - 当`compute`类函数做了实现时，会将compute的返回结果作为`update`的参数传入。
           """
           return acc value
       
       def accumulate(self):
           """
           6. 实现accumulate方法，返回历史batch训练积累后计算得到的评价指标值。
           每次`update`调用时进行数据积累，`accumulate`计算时对积累的所有数据进行计算并返回。
           结算结果会在`fit`接口的训练日志中呈现。
           """
           # 利用update中积累的成员变量数据进行计算后返回
           return accumulated acc value

       def reset(self):
           """
           7. 实现reset方法，每个Epoch结束后进行评估指标的重置，这样下个Epoch可以重新进行计算。
           """
           # do reset action

我们看一个框架中的具体例子，这个是框架中已提供的一个评估指标计算接口，这里就是按照上述说明中的实现方法进行了相关类继承和成员函数实现。

.. code:: python

   from paddle.metric import Metric


   class Precision(Metric):
       """
       Precision (also called positive predictive value) is the fraction of
       relevant instances among the retrieved instances. Refer to
       https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers

       Noted that this class manages the precision score only for binary
       classification task.
       
       ......

       """

       def __init__(self, name='precision', *args, **kwargs):
           super(Precision, self).__init__(*args, **kwargs)
           self.tp = 0  # true positive
           self.fp = 0  # false positive
           self._name = name

       def update(self, preds, labels):
           """
           Update the states based on the current mini-batch prediction results.

           Args:
               preds (numpy.ndarray): The prediction result, usually the output
                   of two-class sigmoid function. It should be a vector (column
                   vector or row vector) with data type: 'float64' or 'float32'.
               labels (numpy.ndarray): The ground truth (labels),
                   the shape should keep the same as preds.
                   The data type is 'int32' or 'int64'.
           """
           if isinstance(preds, paddle.Tensor):
               preds = preds.numpy()
           elif not _is_numpy_(preds):
               raise ValueError("The 'preds' must be a numpy ndarray or Tensor.")

           if isinstance(labels, paddle.Tensor):
               labels = labels.numpy()
           elif not _is_numpy_(labels):
               raise ValueError("The 'labels' must be a numpy ndarray or Tensor.")

           sample_num = labels.shape[0]
           preds = np.floor(preds + 0.5).astype("int32")

           for i in range(sample_num):
               pred = preds[i]
               label = labels[i]
               if pred == 1:
                   if pred == label:
                       self.tp += 1
                   else:
                       self.fp += 1

       def reset(self):
           """
           Resets all of the metric state.
           """
           self.tp = 0
           self.fp = 0

       def accumulate(self):
           """
           Calculate the final precision.

           Returns:
               A scaler float: results of the calculated precision.
           """
           ap = self.tp + self.fp
           return float(self.tp) / ap if ap != 0 else .0

       def name(self):
           """
           Returns metric name
           """
           return self._name

5.5 自定义Callback
~~~~~~~~~~~~~~~~~~

``fit``\ 接口的callback参数支持我们传一个Callback类实例，用来在每轮训练和每个batch训练前后进行调用，可以通过callback收集到训练过程中的一些数据和参数，或者实现一些自定义操作。

.. code:: python

   class SelfDefineCallback(paddle.callbacks.Callback):
       """
       1. 继承paddle.callbacks.Callback
       2. 按照自己的需求实现以下类成员方法：
           def on_train_begin(self, logs=None)                 训练开始前，`Model.fit`接口中调用
           def on_train_end(self, logs=None)                   训练结束后，`Model.fit`接口中调用
           def on_eval_begin(self, logs=None)                  评估开始前，`Model.evaluate`接口调用
           def on_eval_end(self, logs=None)                    评估结束后，`Model.evaluate`接口调用
           def on_test_begin(self, logs=None)                  预测测试开始前，`Model.predict`接口中调用
           def on_test_end(self, logs=None)                    预测测试结束后，`Model.predict`接口中调用 
           def on_epoch_begin(self, epoch, logs=None)          每轮训练开始前，`Model.fit`接口中调用 
           def on_epoch_end(self, epoch, logs=None)            每轮训练结束后，`Model.fit`接口中调用 
           def on_train_batch_begin(self, step, logs=None)     单个Batch训练开始前，`Model.fit`和`Model.train_batch`接口中调用
           def on_train_batch_end(self, step, logs=None)       单个Batch训练结束后，`Model.fit`和`Model.train_batch`接口中调用
           def on_eval_batch_begin(self, step, logs=None)      单个Batch评估开始前，`Model.evalute`和`Model.eval_batch`接口中调用
           def on_eval_batch_end(self, step, logs=None)        单个Batch评估结束后，`Model.evalute`和`Model.eval_batch`接口中调用
           def on_test_batch_begin(self, step, logs=None)      单个Batch预测测试开始前，`Model.predict`和`Model.test_batch`接口中调用
           def on_test_batch_end(self, step, logs=None)        单个Batch预测测试结束后，`Model.predict`和`Model.test_batch`接口中调用
       """
       def __init__(self):
           super(SelfDefineCallback, self).__init__()

       按照需求定义自己的类成员方法

我们看一个框架中的实际例子，这是一个框架自带的ModelCheckpoint回调函数，方便用户在fit训练模型时自动存储每轮训练得到的模型。

.. code:: python

   class ModelCheckpoint(Callback):
       def __init__(self, save_freq=1, save_dir=None):
           self.save_freq = save_freq
           self.save_dir = save_dir

       def on_epoch_begin(self, epoch=None, logs=None):
           self.epoch = epoch

       def _is_save(self):
           return self.model and self.save_dir and ParallelEnv().local_rank == 0

       def on_epoch_end(self, epoch, logs=None):
           if self._is_save() and self.epoch % self.save_freq == 0:
               path = '{}/{}'.format(self.save_dir, epoch)
               print('save checkpoint at {}'.format(os.path.abspath(path)))
               self.model.save(path)

       def on_train_end(self, logs=None):
           if self._is_save():
               path = '{}/final'.format(self.save_dir)
               print('save checkpoint at {}'.format(os.path.abspath(path)))
               self.model.save(path)

6. 模型评估
-----------

对于训练好的模型进行评估操作可以使用\ ``evaluate``\ 接口来实现，事先定义好用于评估使用的数据集后，可以简单的调用\ ``evaluate``\ 接口即可完成模型评估操作，结束后根据prepare中loss和metric的定义来进行相关评估结果计算返回。

返回格式是一个字典： \* 只包含loss，\ ``{'loss': xxx}`` \*
包含loss和一个评估指标，\ ``{'loss': xxx, 'metric name': xxx}`` \*
包含loss和多个评估指标，\ ``{'loss': xxx, 'metric name': xxx, 'metric name': xxx}``

.. code:: ipython3

    result = model.evaluate(val_dataset, verbose=1)


.. parsed-literal::

    Eval begin...
    step 10000/10000 [==============================] - loss: 0.0000e+00 - acc: 0.9801 - 2ms/step          
    Eval samples: 10000


7. 模型预测
-----------

高层API中提供了\ ``predict``\ 接口来方便用户对训练好的模型进行预测验证，只需要基于训练好的模型将需要进行预测测试的数据放到接口中进行计算即可，接口会将经过模型计算得到的预测结果进行返回。

返回格式是一个list，元素数目对应模型的输出数目： \*
模型是单一输出：[(numpy_ndarray_1, numpy_ndarray_2, …, numpy_ndarray_n)]
\* 模型是多输出：[(numpy_ndarray_1, numpy_ndarray_2, …,
numpy_ndarray_n), (numpy_ndarray_1, numpy_ndarray_2, …,
numpy_ndarray_n), …]

numpy_ndarray_n是对应原始数据经过模型计算后得到的预测数据，数目对应预测数据集的数目。

.. code:: ipython3

    pred_result = model.predict(val_dataset)


.. parsed-literal::

    Predict begin...
    step 10000/10000 [==============================] - 4ms/step          
    Predict samples: 10000


7.1 使用多卡进行预测
~~~~~~~~~~~~~~~~~~~~

有时我们需要进行预测验证的数据较多，单卡无法满足我们的时间诉求，那么\ ``predict``\ 接口也为用户支持实现了使用多卡模式来运行。

使用起来也是超级简单，无需修改代码程序，只需要使用launch来启动对应的预测脚本即可。

.. code:: bash

   $ python3 -m paddle.distributed.launch infer.py

infer.py里面就是包含model.predict的代码程序。

8. 模型部署
-----------

8.1 模型存储
~~~~~~~~~~~~

模型训练和验证达到我们的预期后，可以使用\ ``save``\ 接口来将我们的模型保存下来，用于后续模型的Fine-tuning（接口参数training=True）或推理部署（接口参数training=False）。

需要注意的是，在动态图模式训练时保存推理模型的参数文件和模型文件，需要在forward成员函数上添加@paddle.jit.to_static装饰器，参考下面的例子：

.. code:: python

   class Mnist(paddle.nn.Layer):
       def __init__(self):
           super(Mnist, self).__init__()

           self.flatten = paddle.nn.Flatten()
           self.linear_1 = paddle.nn.Linear(784, 512)
           self.linear_2 = paddle.nn.Linear(512, 10)
           self.relu = paddle.nn.ReLU()
           self.dropout = paddle.nn.Dropout(0.2)

       @paddle.jit.to_static
       def forward(self, inputs):
           y = self.flatten(inputs)
           y = self.linear_1(y)
           y = self.relu(y)
           y = self.dropout(y)
           y = self.linear_2(y)

           return y

.. code:: ipython3

    model.save('~/model/mnist')

8.2 预测部署
~~~~~~~~~~~~

有了用于推理部署的模型，就可以使用推理部署框架来完成预测服务部署，具体可以参见：\ `预测部署 <https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/index_cn.html>`__\ ，
包括服务端部署、移动端部署和模型压缩。
