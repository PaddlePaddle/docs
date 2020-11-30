=============
API Reference
=============



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


..  toctree::
    :maxdepth: 1

    ../api_guides/index_cn.rst
    paddle_cn.rst
    dataset_cn.rst
    tensor_cn.rst
    nn_cn.rst
    imperative_cn.rst
    declarative_cn.rst
    optimizer_cn.rst
    static_cn.rst
    metric_cn.rst
    framework_cn.rst
    io_cn.rst
    utils_cn.rst
    incubate_cn.rst
    fluid_cn.rst
    backward_cn.rst
    clip_cn.rst
    data_cn/data_reader_cn.rst
    data_cn/dataset_cn.rst
    dataset_cn.rst
    distributed_cn.rst
    dygraph_cn.rst
    executor_cn.rst
    initializer_cn.rst
    io_cn.rst
    layers_cn.rst
    metrics_cn.rst
    nets_cn.rst
    profiler_cn.rst
    regularizer_cn.rst
    transpiler_cn.rst
    unique_name_cn.rst
    static_cn.rst
