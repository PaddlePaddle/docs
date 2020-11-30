快速上手
===========

飞桨2.0概述
-----------
在保持1.x版本工业级大规模高效训练和多平台快速部署优势的前提，飞桨2.0版本重点提升了框架的易用性，主要在用户交互层进行了优化，降低学习门槛，提升开发效率。不管对于初学者还是资深专家，都能够更容易地使用飞桨进行深度学习任务开发，加速前沿算法研究和工业级任务开发。

此版本为测试版，还在迭代开发中，目前还没有稳定，后续API会根据反馈有可能进行不兼容的升级。对于想要体验飞桨最新特性的开发者，欢迎试用此版本；对稳定性要求高的工业级应用场景推荐使用Paddle
1.8稳定版本。此版本主推命令式(imperative)开发模式，并提供了高层API的封装。命令式开发模式具有很好的灵活性，高层API可以大幅减少重复代码。对于初学者或基础的任务场景，推荐使用高层API的开发方式，简单易用；对于资深开发者想要实现复杂的功能，推荐使用动态图的API，灵活高效。

跟1.x版本对比，飞桨2.0版本的重要升级如下：

+------------+--------------------------------------+-----------------------------------------+
|            | 飞桨1.x版本                          | 飞桨2.0版本                             |
+============+======================================+=========================================+
| 开发模式   | 推荐声明式（declarative)             | 推荐命令式(imperative)                  |
+------------+--------------------------------------+-----------------------------------------+
| 组网方式   | 推荐函数式组网                       | 推荐面向对象式组网                      |
+------------+--------------------------------------+-----------------------------------------+
| 高层API    | 无                                   | 封装常见的操作，实现低代码开发          |
+------------+--------------------------------------+-----------------------------------------+
| 基础API    | fluid目录，结构不清晰，存在过时API   | paddle目录，整体结构调整，清理废弃API   |
+------------+--------------------------------------+-----------------------------------------+

开发模式
--------

飞桨同时支持声明式和命令式这两种开发模式，兼具声明式编程的高效和命令式编程的灵活。

声明式编程模式（通常也被称为静态模式或define-and-run模式），程序可以明确分为网络结构定义和执行这两个阶段。定义阶段声明网络结构，此时并未传入具体的训练数据；执行阶段需要用户通过feed的方式传入具体数据，完成计算后，通过fetch的方式返回计算结果。示例如下：

.. code:: python

    import numpy
    import paddle
    # 定义输入数据占位符
    a = paddle.static.data(name="a", shape=[1], dtype='int64')
    b = paddle.static.data(name="b", shape=[1], dtype='int64')
    # 组建网络（此处网络仅由一个操作构成，即elementwise_add）
    result = paddle.elementwise_add(a, b)
    # 准备运行网络
    cpu = paddle.CPUPlace() # 定义运算设备，这里选择在CPU下训练
    exe = paddle.Executor(cpu) # 创建执行器
    # 创建输入数据
    x = numpy.array([2])
    y = numpy.array([3])
    # 运行网络
    outs = exe.run(
        feed={'a':x, 'b':y}, # 将输入数据x, y分别赋值给变量a，b
        fetch_list=[result]  # 通过fetch_list参数指定需要获取的变量结果
        )
    #输出运行结果
    print (outs)
    #[array([5], dtype=int64)]

声明式开发模式的优点为在程序执行之前，可以拿到全局的组网信息，方便对计算图进行全局的优化，提升性能；并且由于全局计算图的存在，方便将计算图导出到文件，方便部署到非python语言的开发环境中，比如：C/C++/JavaScript等。声明式开发模式的缺点为，由于网络定义和执行阶段分离，在定义的时候并不知道所执行的具体的数据，程序的开发和调试会比较困难。

命令式编程模式（通常也被称为动态模式、eager模式或define-by-run模式），程序在网络结构定义的同时立即执行，能够实时的到执行结果。示例如下：

.. code:: python

    import numpy
    import paddle
    from paddle.imperative import to_variable

    # 切换命令式编程模式
    paddle.enable_imperative()

    # 创建数据
    x = to_variable(numpy.array([2]))
    y = to_variable(numpy.array([3]))
    # 定义运算并执行
    z = paddle.elementwise_add(x, y)
    # 输出执行结果
    print (z.numpy())

飞桨2.0推荐开发者使用命令式编程，可以使用原生python控制流API，具有灵活，容易开发调试的优点；同时为了兼具声明式编程在性能和部署方面的优势，飞桨提供了自动转换功能，可以将包含python控制流的代码，转换为Program，通过底层的Executor进行执行。

组网方式
--------

飞桨1.x大量使用函数式的组网方式，这种方法的好处是写法很简洁，但表达能力偏弱，比如：如果我们想要查看fc隐含的参数的值或者想要对某一个参数进行裁剪时，会很困难，我们需要操作隐含的参数名才能访问。比如：

.. code:: python

    import paddle.fluid as fluid

    data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
    fc = fluid.layers.fc(input=data, size=1000, act="tanh")

飞桨2.0推荐使用面向对象式的组网方式，需要通过继承\ ``paddle.nn.Layer``\ 类的\ ``__init__``\ 和\ ``forward``\ 函数实现网络结构自定义，这种方式通过类的成员变量，方便地访问到每个类的成员，比如：

.. code:: python

    import paddle

    class SimpleNet(paddle.nn.Layer):
        def __init__(self, in_size, out_size):
            super(SimpleNet, self).__init__()
            self._linear = paddle.nn.Linear(in_size, out_size)

        def forward(self, x):
            y = self._linear(x)
            return y

高层API
-------

使用飞桨进行深度学习任务的开发，整体过程包括数据处理、组网、训练、评估、模型导出、预测部署这些基本的操作。这些基本操作在不同的任务中会反复出现，使用基础API进行开发时，需要开发者重复地写这些基础操作的代码，增加了模型开发的工作量。高层API针对这些基础操作进行了封装，提供更高层的开发接口，开发者只需要关心数据处理和自定义组网，其他工作可以通过调用高层API来完成。在MNIST手写数字识别任务中，对比动态图基础API的实现方式，通过使用高层API可以减少80%的非组网类代码。

使用高层API的另外一个好处是，可以通过一行代码\ ``paddle.enable_imperative``\ ，切换命令式编程模式和声明式编程模式。在开发阶段，可以使用的命令式编程模式，方便调试；开发完成后，可以切换到声明式编程模式，加速训练和方便部署。兼具了命令式编程实时执行，容易调试的优点，以及声明式编程全局优化和容易部署的优点。

以下为高层API的一个基础示例

.. code:: python

    import numpy as np
    import paddle
    import paddle.nn.functional as F
    from paddle.incubate.hapi.model import Model, Input, Loss
    from paddle.incubate.hapi.loss import CrossEntropy

    #高层API的组网方式需要继承Model，Model类实现了模型执行所需的逻辑
    class SimpleNet(Model):
        def __init__(self, in_size, out_size):
            super(SimpleNet, self).__init__()
            self._linear = paddle.nn.Linear(in_size, out_size)
        def forward(self, x):
            y = self._linear(x)
            z = self._linear(y)
            pred = F.softmax(z)
            return pred

    #兼容声明式开发模式，定义数据形状类型，如果不使用声明式编程模式，可以不定义数据占位符
    inputs = [Input([None, 8], 'float32', name='image')]
    labels = [Input([None, 1], 'int64', name='labels')]

    #定义模型网络结构，包括指定损失函数和优化算法
    model = SimpleNet(8, 8)
    optimizer = paddle.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=model.parameters())
    model.prepare(optimizer, CrossEntropy(), None, inputs, labels, device='cpu')

    #切换执行模式
    paddle.enable_imperative(paddle.CPUPlace())

    #基于batch的训练
    batch_num = 10
    x = np.random.random((4, 8)).astype('float32')
    y = np.random.randint(0, 8, (4, 1)).astype('int64')
    for i in range(batch_num):
        model.train_batch(inputs=x, labels=y)

更多高层API开发的模型和示例请参考github Repo:
`hapi <https://github.com/paddlepaddle/hapi>`__

基础API
-------

飞桨2.0提供了新的API，可以同时支持声明式和命令式两种开发模式，比如paddle.nn.Linear，避免在两种模式下使用不同的API造成困惑。原飞桨1.x的API位于paddle.fluid目录下，其中部分组网类的API，只能用于声明式开发，比如：fluid.layers.fc，无法用于命令式开发。

飞桨2.0对API的目录结构进行了调整，从原来的paddle.fluid目录调整到paddle目录下，使得开发接口更加清晰，调整后的目录结构如下：

+---------------------+-----------------------------------------------------------------------------------------------------------+
| 目录                | 功能和包含API                                                                                             |
+=====================+===========================================================================================================+
| paddle.\*           | paddle根目录下保留了常用API的别名，当前包括：paddle.tensor, paddle.framework目录下的所有API               |
+---------------------+-----------------------------------------------------------------------------------------------------------+
| paddle.tensor       | 跟tensor操作相关的API，比如：创建zeros, 矩阵运算matmul, 变换concat, 计算elementwise\_add, 查找argmax等    |
+---------------------+-----------------------------------------------------------------------------------------------------------+
| paddle.nn           | 跟组网相关的API，比如：输入占位符data/Input，控制流while\_loop/cond，损失函数，卷积，LSTM等，激活函数等   |
+---------------------+-----------------------------------------------------------------------------------------------------------+
| paddle.framework    | 基础框架相关的API，比如：Variable, Program, Executor等                                                    |
+---------------------+-----------------------------------------------------------------------------------------------------------+
| paddle.imperative   | imprerative模式专用的API，比如：to\_variable, prepare\_context等                                          |
+---------------------+-----------------------------------------------------------------------------------------------------------+
| paddle.optimizer    | 优化算法相关API，比如：SGD，Adagrad, Adam等                                                               |
+---------------------+-----------------------------------------------------------------------------------------------------------+
| paddle.metric       | 评估指标计算相关的API，比如：accuracy, cos\_sim等                                                         |
+---------------------+-----------------------------------------------------------------------------------------------------------+
| paddle.io           | 数据输入输出相关API，比如：save, load, Dataset, DataLoader等                                              |
+---------------------+-----------------------------------------------------------------------------------------------------------+
| paddle.device       | 设备管理相关API，比如：CPUPlace， CUDAPlace等                                                             |
+---------------------+-----------------------------------------------------------------------------------------------------------+
| paddle.fleet        | 分布式相关API                                                                                             |
+---------------------+-----------------------------------------------------------------------------------------------------------+

同时飞桨2.0对部分Paddle
1.x版本的API进行了清理，删除了部分不再推荐使用的API，具体信息请参考Release
Note。

..  toctree::
    :hidden:

    basic_concept/index_cn.rst
    dygraph/DyGraph.md
    hapi.md

