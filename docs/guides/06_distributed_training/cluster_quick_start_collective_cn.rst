数据并行(data parallelism)是大规模深度学习训练中常用的并行模式，它将训练任务切分到多个进程(设备)上运行，其中每个进程维护相同的模型参数和相同的计算任务，但处理不同的数据(batch data)。通过这种方式，同一全局数据(global batch)下的数据和计算被切分到了不同的进程，从而减轻了单个设备上的计算和存储压力。
本节将采用自定义卷积网络和Paddle内置的CIFAR-10数据集来介绍如何使用 `Fleet API <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/Overview_cn.html#fleetapi>`_ (paddle.distributed.fleet) 进行数据并行训练。

1.1 版本要求
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在编写分布式训练程序之前，用户需要确保已经安装paddlepaddle-2.0.0-rc-cpu或paddlepaddle-2.0.0-rc-gpu及以上版本的飞桨开源框架。

1.2 具体步骤 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

与单机单卡的普通模型训练相比，数据并行训练只需要按照如下5个步骤对代码进行简单调整即可：

    1. 导入分布式训练依赖包 
    2. 初始化Fleet环境 
    3. 构建分布式训练使用的网络模型 
    4. 构建分布式训练使用的优化器 
    5. 构建分布式训练使用的数据加载器 

下面将逐一进行讲解。

1.2.1 导入分布式训练依赖包
""""""""""""""""""""""""""""

导入飞桨分布式训练专用包Fleet。

.. code-block:: python

    # 导入分布式专用Fleet API
    from paddle.distributed import fleet
    # 导入分布式训练数据所需API
    from paddle.io import DataLoader, DistributedBatchSampler
    # 设置GPU环境
    paddle.set_device('gpu')

1.2.2 初始化Fleet环境
""""""""""""""""""""""""""

分布式初始化需要：

    1. 设置is_collective为True，表示分布式训练采用Collective模式。
    2. [可选] 设置分布式策略 `DistributedStrategy <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/fleet/DistributedStrategy_cn.html>`_，跳过将使用缺省配置。

.. code-block:: python

    # 选择不设置分布式策略
    fleet.init(is_collective=True)

    # 选择设置分布式策略
    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)

1.2.3 构建分布式训练使用的网络模型
""""""""""""""""""""""""""""""""""
只需要使用 ``fleet.distributed_model`` 对原始串行网络模型进行封装。

.. code-block:: python

    # 等号右边model为原始串行网络模型
    model = fleet.distributed_model(model)

1.2.4 构建分布式训练使用的优化器
""""""""""""""""""""""""""""""""""
只需要使用 ``fleet.distributed_optimizer`` 对原始串行优化器进行封装。

.. code-block:: python

    # 等号右边optimizer为原始串行网络模型
    optimizer = fleet.distributed_optimizer(optimizer)

1.2.5 构建分布式训练使用的数据加载器
"""""""""""""""""""""""""""""""""""""""""""""

由于分布式训练过程中每个进程可能读取不同数据，所以需要对数据集进行合理拆分后再进行加载。这里只需要在构建 `DataLoader <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader>`_ 时, 设置分布式数据采样器 `DistributedBatchSampler <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DistributedBatchSampler_cn.html#distributedbatchsampler>`_ 即可。

.. code-block:: python

    # 构建分布式数据采样器 
    # 注意：需要保证batch中每个样本数据shape相同，若原尺寸不一，需进行预处理
    train_sampler = DistributedBatchSampler(train_dataset, 32, shuffle=True, drop_last=True)
    val_sampler = DistributedBatchSampler(val_dataset, 32)

    # 构建分布式数据加载器
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)
    valid_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=2)

1.3 完整示例代码
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # -*- coding: UTF-8 -*-
    import numpy as np
    import matplotlib.pyplot as plt
    import paddle
    import paddle.nn.functional as F
    from paddle.vision.transforms import ToTensor
    # 一、导入分布式专用Fleet API
    from paddle.distributed import fleet
    # 构建分布式数据加载器所需API
    from paddle.io import DataLoader, DistributedBatchSampler
    # 设置GPU环境
    paddle.set_device('gpu')

    class MyNet(paddle.nn.Layer):
        def __init__(self, num_classes=1):
            super(MyNet, self).__init__()

            self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3))
            self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

            self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=64, kernel_size=(3,3))
            self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

            self.conv3 = paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3,3))

            self.flatten = paddle.nn.Flatten()

            self.linear1 = paddle.nn.Linear(in_features=1024, out_features=64)
            self.linear2 = paddle.nn.Linear(in_features=64, out_features=num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)

            x = self.conv3(x)
            x = F.relu(x)

            x = self.flatten(x)
            x = self.linear1(x)
            x = F.relu(x)
            x = self.linear2(x)
            return x

    epoch_num = 10
    batch_size = 32
    learning_rate = 0.001
    val_acc_history = []
    val_loss_history = []

    def train():
        # 二、初始化Fleet环境
        fleet.init(is_collective=True)

        model = MyNet(num_classes=10)
        # 三、构建分布式训练使用的网络模型
        model = fleet.distributed_model(model)

        opt = paddle.optimizer.Adam(learning_rate=learning_rate,parameters=model.parameters())
        # 四、构建分布式训练使用的优化器
        opt = fleet.distributed_optimizer(opt)

        transform = ToTensor()
        cifar10_train = paddle.vision.datasets.Cifar10(mode='train',
                                               transform=transform)
        cifar10_test = paddle.vision.datasets.Cifar10(mode='test',
                                              transform=transform)

        # 五、构建分布式训练使用的数据集
        train_sampler = DistributedBatchSampler(cifar10_train, 32, shuffle=True, drop_last=True)
        train_loader = DataLoader(cifar10_train, batch_sampler=train_sampler, num_workers=2)

        valid_sampler = DistributedBatchSampler(cifar10_test, 32, drop_last=True)
        valid_loader = DataLoader(cifar10_test, batch_sampler=valid_sampler, num_workers=2)


        for epoch in range(epoch_num):
            model.train()
            for batch_id, data in enumerate(train_loader()):
                x_data = data[0]
                y_data = paddle.to_tensor(data[1])
                y_data = paddle.unsqueeze(y_data, 1)

                logits = model(x_data)
                loss = F.cross_entropy(logits, y_data)

                if batch_id % 1000 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
                loss.backward()
                opt.step()
                opt.clear_grad()

            model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(valid_loader()):
                x_data = data[0]
                y_data = paddle.to_tensor(data[1])
                y_data = paddle.unsqueeze(y_data, 1)

                logits = model(x_data)
                loss = F.cross_entropy(logits, y_data)
                acc = paddle.metric.accuracy(logits, y_data)
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())

            avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
            print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
            val_acc_history.append(avg_acc)
            val_loss_history.append(avg_loss)

    if __name__ == "__main__":
        train()


1.4 分布式启动
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

准备好分布式训练脚本后，就可以通过 `paddle.distributed.launch <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/launch_cn.html#launch>`_ 在集群上启动分布式训练：

- 多机多卡训练
    假设集群包含两个节点，每个节点上可使用的GPU卡数为4，IP地址分别为192.168.1.2和192.168.1.3，那么需要在两个节点的终端上分别运行如下命令：

    在192.168.1.2节点运行：
    
        .. code-block:: bash

            python -m paddle.distributed.launch \
            --cluster_node_ips=192.168.1.2,192.168.1.3 \
            --node_ip=192.168.1.2 \
            --started_port=6170 \
            --selected_gpus=0,1,2,3 \
            train_with_fleet.py

    在192.168.1.3节点运行：

        .. code-block:: bash

            python -m paddle.distributed.launch \
            --cluster_node_ips=192.168.1.2,192.168.1.3 \
            --node_ip=192.168.1.3 \
            --started_port=6170 \
            --selected_gpus=0,1,2,3 \
            train_with_fleet.py


- 单机多卡训练
    假设只使用集群的一个节点，节点上可使用的GPU卡数为4，那么只需要在节点终端运行如下命令：

    .. code-block:: bash

        python -m paddle.distributed.launch --selected_gpus=0,1,2,3 train_with_fleet.py
