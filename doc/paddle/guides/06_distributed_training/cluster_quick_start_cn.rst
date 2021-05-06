..  _cluster_quick_start:

分布式训练快速开始
==================

一、Collective 训练快速开始
-------------------------

本节将采用CV领域非常经典的模型ResNet50为例，介绍如何使用Fleet API（paddle.distributed.fleet）完成Collective训练任务。 数据方面我们采用Paddle内置的flowers数据集，优化器使用Momentum方法。循环迭代多个epoch，每轮打印当前网络具体的损失值和acc值。 具体代码保存在FleetX/examples/resnet下面， 其中包含动态图和静态图两种执行方式。resnet_dygraph.py为动态图模型相关代码，train_fleet_dygraph.py为动态图训练脚本。 resnet_static.py为静态图模型相关代码，而train_fleet_static.py为静态图训练脚本。

1.1 版本要求
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在编写分布式训练程序之前，用户需要确保已经安装paddlepaddle-2.0.0-rc-cpu或paddlepaddle-2.0.0-rc-gpu及以上版本的飞桨开源框架。

1.2 操作方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

与单机单卡的普通模型训练相比，无论静态图还是动态图，Collective训练的代码都只需要补充三个部分代码：

    1. 导入分布式训练需要的依赖包。
    2. 初始化Fleet环境。
    3. 设置分布式训练需要的优化器。 

下面将逐一进行讲解。

1.2.1 导入依赖
""""""""""""

导入必要的依赖，例如分布式训练专用的Fleet API(paddle.distributed.fleet)。

.. code-block:: python

    from paddle.distributed import fleet

1.2.2 初始化fleet环境
""""""""""""

包括定义缺省的分布式策略，然后通过将参数is_collective设置为True，使训练架构设定为Collective架构。

.. code-block:: python

    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)

1.2.3 设置分布式训练使用的优化器
""""""""""""

使用distributed_optimizer设置分布式训练优化器。

.. code-block:: python

    optimizer = fleet.distributed_optimizer(optimizer)

1.3 动态图完整代码
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

train_fleet_dygraph.py的完整训练代码如下所示。

.. code-block:: python

    # -*- coding: UTF-8 -*-
    import numpy as np
    import argparse
    import ast
    import paddle
    # 导入必要分布式训练的依赖包
    from paddle.distributed import fleet
    # 导入模型文件
    from resnet_dygraph import ResNet

    base_lr = 0.1   # 学习率
    momentum_rate = 0.9 # 冲量
    l2_decay = 1e-4 # 权重衰减

    epoch = 10  #训练迭代次数
    batch_size = 32 #训练批次大小
    class_dim = 102

    # 设置数据读取器
    def reader_decorator(reader):
        def __reader__():
            for item in reader():
                img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield img, label

        return __reader__

    # 设置优化器
    def optimizer_setting(parameter_list=None):
        optimizer = paddle.optimizer.Momentum(
            learning_rate=base_lr,
            momentum=momentum_rate,
            weight_decay=paddle.regularizer.L2Decay(l2_decay),
            parameters=parameter_list)
        return optimizer

    # 设置训练函数
    def train_resnet():
        # 初始化Fleet环境
        fleet.init(is_collective=True)

        resnet = ResNet(class_dim=class_dim, layers=50)

        optimizer = optimizer_setting(parameter_list=resnet.parameters())
        optimizer = fleet.distributed_optimizer(optimizer)
        # 通过Fleet API获取分布式model，用于支持分布式训练
        resnet = fleet.distributed_model(resnet)

        train_reader = paddle.batch(
                reader_decorator(paddle.dataset.flowers.train(use_xmap=True)),
                batch_size=batch_size,
                drop_last=True)

        train_loader = paddle.io.DataLoader.from_generator(
            capacity=32,
            use_double_buffer=True,
            iterable=True,
            return_list=True,
            use_multiprocess=True)
        train_loader.set_sample_list_generator(train_reader)

        for eop in range(epoch):
            resnet.train()

            for batch_id, data in enumerate(train_loader()):
                img, label = data
                label.stop_gradient = True

                out = resnet(img)
                loss = paddle.nn.functional.cross_entropy(input=out, label=label)
                avg_loss = paddle.mean(x=loss)
                acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
                acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

                dy_out = avg_loss.numpy()

                avg_loss.backward()

                optimizer.minimize(avg_loss)
                resnet.clear_gradients()
                if batch_id % 5 == 0:
                    print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, dy_out, acc_top1, acc_top5))
    # 启动训练
    if __name__ == '__main__':
        train_resnet()

1.4 静态图完整代码
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

train_fleet_static.py的完整训练代码如下所示。

.. code-block:: python

    # -*- coding: UTF-8 -*-
    import numpy as np
    import argparse
    import ast
    import paddle
    # 导入必要分布式训练的依赖包
    import paddle.distributed.fleet as fleet
    # 导入模型文件
    import resnet_static as resnet
    import os

    base_lr = 0.1   # 学习率
    momentum_rate = 0.9 # 冲量
    l2_decay = 1e-4 # 权重衰减

    epoch = 10  #训练迭代次数
    batch_size = 32 #训练批次大小
    class_dim = 10

    # 设置优化器
    def optimizer_setting(parameter_list=None):
        optimizer = paddle.optimizer.Momentum(
            learning_rate=base_lr,
            momentum=momentum_rate,
            weight_decay=paddle.regularizer.L2Decay(l2_decay),
            parameters=parameter_list)
        return optimizer
    # 设置数据读取器
    def get_train_loader(feed_list, place):
        def reader_decorator(reader):
            def __reader__():
                for item in reader():
                    img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
                    label = np.array(item[1]).astype('int64').reshape(1)
                    yield img, label

            return __reader__
        train_reader = paddle.batch(
                reader_decorator(paddle.dataset.flowers.train(use_xmap=True)),
                batch_size=batch_size,
                drop_last=True)
        train_loader = paddle.io.DataLoader.from_generator(
            capacity=32,
            use_double_buffer=True,
            feed_list=feed_list,
            iterable=True)
        train_loader.set_sample_list_generator(train_reader, place)
        return train_loader
    # 设置训练函数
    def train_resnet():
        paddle.enable_static() # 使能静态图功能
        paddle.vision.set_image_backend('cv2')

        image = paddle.static.data(name="x", shape=[None, 3, 224, 224], dtype='float32')
        label= paddle.static.data(name="y", shape=[None, 1], dtype='int64')
        # 调用ResNet50模型
        model = resnet.ResNet(layers=50)
        out = model.net(input=image, class_dim=class_dim)
        avg_cost = paddle.nn.functional.cross_entropy(input=out, label=label)
        acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
        acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
        # 设置训练资源，本例使用GPU资源
        place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))

        train_loader = get_train_loader([image, label], place)
        #初始化Fleet环境
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)
        optimizer = optimizer_setting()

        # 通过Fleet API获取分布式优化器，将参数传入飞桨的基础优化器
        optimizer = fleet.distributed_optimizer(optimizer)
        optimizer.minimize(avg_cost)

        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        epoch = 10
        step = 0
        for eop in range(epoch):
            for batch_id, data in enumerate(train_loader()):
                loss, acc1, acc5 = exe.run(paddle.static.default_main_program(), feed=data, fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])
                if batch_id % 5 == 0:
                    print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, loss, acc1, acc5))
    # 启动训练
    if __name__ == '__main__':
        train_resnet()

1.5 运行示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

假设要运行2卡的任务，那么只需在命令行中执行:

动态图：

.. code-block:: bash

    python -m paddle.distributed.launch --gpus=0,1 train_fleet_dygraph.py


您将看到显示如下日志信息：

.. code-block:: bash

    -----------  Configuration Arguments -----------
    gpus: 0,1
    heter_worker_num: None
    heter_workers:
    http_port: None
    ips: 127.0.0.1
    log_dir: log
    ...
    ------------------------------------------------
    launch train in GPU mode
    INFO 2021-03-23 14:11:38,107 launch_utils.py:481] Local start 2 processes. First process distributed environment info (Only For Debug):
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:59648               |
        |                     PADDLE_TRAINERS_NUM                        2                      |
        |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:59648,127.0.0.1:50871       |
        |                     FLAGS_selected_gpus                        0                      |
        |                       PADDLE_TRAINER_ID                        0                      |
        +=======================================================================================+

    I0323 14:11:39.383992  3788 nccl_context.cc:66] init nccl context nranks: 2 local rank: 0 gpu id: 0 ring id: 0
    W0323 14:11:39.872674  3788 device_context.cc:368] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
    W0323 14:11:39.877283  3788 device_context.cc:386] device: 0, cuDNN Version: 7.4.
    [Epoch 0, batch 0] loss: 4.77086, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 5] loss: 15.69098, acc1: 0.03125, acc5: 0.18750
    [Epoch 0, batch 10] loss: 23.41379, acc1: 0.00000, acc5: 0.09375
    ...


静态图：

.. code-block:: bash

    python -m paddle.distributed.launch --gpus=0,1 train_fleet_static.py


您将看到显示如下日志信息：

.. code-block:: bash

    -----------  Configuration Arguments -----------
    gpus: 0,1
    heter_worker_num: None
    heter_workers:
    http_port: None
    ips: 127.0.0.1
    log_dir: log
    ...
    ------------------------------------------------
    WARNING 2021-01-04 17:59:08,725 launch.py:314] Not found distinct arguments and compiled with cuda. Default use collective mode
    launch train in GPU mode
    INFO 2021-01-04 17:59:08,727 launch_utils.py:472] Local start 2 processes. First process distributed environment info (Only For Debug):
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:17901               |
        |                     PADDLE_TRAINERS_NUM                        2                      |
        |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:17901,127.0.0.1:18846       |
        |                     FLAGS_selected_gpus                        0                      |
        |                       PADDLE_TRAINER_ID                        0                      |
        +=======================================================================================+

    ...
    W0104 17:59:19.018365 43338 device_context.cc:342] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 9.2
    W0104 17:59:19.022523 43338 device_context.cc:352] device: 0, cuDNN Version: 7.4.
    W0104 17:59:23.193490 43338 fuse_all_reduce_op_pass.cc:78] Find all_reduce operators: 161. To make the speed faster, some all_reduce ops are fused during training, after fusion, the number of all_reduce ops is 5.
    [Epoch 0, batch 0] loss: 0.12432, acc1: 0.00000, acc5: 0.06250
    [Epoch 0, batch 5] loss: 1.01921, acc1: 0.00000, acc5: 0.00000
    ...

从单机多卡到多机多卡训练，在代码上不需要做任何改动，只需再额外指定ips参数即可。其内容为多机的ip列表，命令如下所示：

.. code-block:: bash

    # 动态图
    python -m paddle.distributed.launch --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus 0,1,2,3,4,5,6,7 train_fleet_dygraph.py

    # 静态图
    python -m paddle.distributed.launch --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus 0,1,2,3,4,5,6,7 train_fleet_static.py



二、ParameterServer训练
-------------------------

本节将采用推荐领域非常经典的模型wide_and_deep为例，介绍如何使用Fleet API（paddle.distributed.fleet）完成参数服务器训练任务，本次快速开始的示例代码位于https://github.com/PaddlePaddle/FleetX/tree/develop/examples/wide_and_deep。

2.1 版本要求
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在编写分布式训练程序之前，用户需要确保已经安装paddlepaddle-2.0.0-rc-cpu或paddlepaddle-2.0.0-rc-gpu及以上版本的飞桨开源框架。

2.2 操作方法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

参数服务器训练的基本代码主要包括如下几个部分：

    1. 导入分布式训练需要的依赖包。
    2. 定义分布式模式并初始化分布式训练环境。
    3. 加载模型及数据。
    4. 定义参数更新策略及优化器。
    5. 开始训练。 
    
下面将逐一进行讲解。

2.2.1 导入依赖
""""""""""""

导入必要的依赖，例如分布式训练专用的Fleet API(paddle.distributed.fleet)。

.. code-block:: python

    import paddle
    import paddle.distributed.fleet as fleet
    import paddle.distributed.fleet.base.role_maker as role_maker

2.2.2 定义分布式模式并初始化分布式训练环境
""""""""""""

通过 `fleet.init()` 接口，用户可以定义训练相关的环境，注意此环境是用户预先在环境变量中配置好的，包括：训练节点个数，服务节点个数，当前节点的序号，服务节点完整的IP:PORT列表等。

.. code-block:: python

    # 当前参数服务器模式只支持静态图模式， 因此训练前必须指定`paddle.enable_static()`
    paddle.enable_static()
    role = role_maker.PaddleCloudRoleMaker()
    fleet.init(role)

2.2.3 加载模型及数据
""""""""""""

.. code-block:: python

    # 模型定义参考examples/wide_and_deep中model.py
    from model import net
    from reader import data_reader

    feeds, predict, avg_cost = net()

    train_reader = paddle.batch(data_reader(), batch_size=4)
    reader.decorate_sample_list_generator(train_reader)

2.2.4 定义同步训练 Strategy 及 Optimizer
""""""""""""

在Fleet API中，用户可以使用`fleet.DistributedStrategy()`接口定义自己想要使用的分布式策略。

其中`a_sync`选项用于定义参数服务器相关的策略，当其被设定为`False`时，分布式训练将在同步的模式下进行。反之，当其被设定成`True`时，分布式训练将在异步的模式下进行。

.. code-block:: python

    # 定义异步训练
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = True

    # 定义同步训练
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = False

    # 定义Geo异步训练, Geo异步目前只支持SGD优化算法
    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.a_sync = True
    dist_strategy.a_sync_configs = {"k_steps": 100}

    optimizer = paddle.optimizer.SGD(learning_rate=0.0001)
    optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    optimizer.minimize(model.loss)

2.2.5 开始训练
""""""""""""

完成模型及训练策略以后，我们就可以开始训练模型了。因为在参数服务器模式下会有不同的角色，所以根据不同节点分配不同的任务。

对于服务器节点，首先用`init_server()`接口对其进行初始化，然后启动服务并开始监听由训练节点传来的梯度。

同样对于训练节点，用`init_worker()`接口进行初始化后， 开始执行训练任务。运行`exe.run()`接口开始训练，并得到训练中每一步的损失值。

.. code-block:: python

    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()
    else:
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())

        fleet.init_worker()

        for epoch_id in range(1):
            reader.start()
            try:
                while True:
                    loss_val = exe.run(program=paddle.static.default_main_program(),
                                    fetch_list=[avg_cost.name])
                    loss_val = np.mean(loss_val)
                    print("TRAIN ---> pass: {} loss: {}\n".format(epoch_id,
                                                                loss_val))
            except paddle.core.EOFException:
                reader.reset()

        fleet.stop_worker()

2.3 运行训练脚本
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

定义完训练脚本后，我们就可以用`python -m paddle.distributed.launch`指令运行分布式任务了。其中`server_num`, `worker_num`分别为服务节点和训练节点的数量。在本例中，服务节点有1个，训练节点有2个。

.. code-block:: bash

    python -m paddle.distributed.launch --server_num=1 --worker_num=2 train.py