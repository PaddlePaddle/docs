..  _data_parallel:

数据并行使用手册
=========

数据并行是深度学习领域最常用的并行方法。如下图(a)所示，在此策略下数据集被平均分为多份，每个卡上保存完整的模型参数并独立处理一份子数据集，以加速模型训练过程。

.. image:: ./img/data_parallel.png
  :width: 800
  :alt: Data Parallel
  :align: center

一、原理介绍
-----------------------

深度学习模型训练过程计算通常分为前向计算、反向计算和梯度更新。由于各个计算设备的初始随机状态不同，各个计算设备上的初始模型参数也因此存在差异。数据并行方式下，为了保持各个计算设备上参数的一致性，在初始阶段需要通过广播的方式将第一张计算设备上的模型参数广播到其它所有计算设备。这样，各个计算设备上的模型参数在广播完成后是一致的。前向计算阶段，各个计算设备使用自己的数据计算模型损失值。由于各个计算设备读取的数据不同，因此各个计算设备上得到的模型损失值也往往是不同的。反向计算阶段，各个计算设备根据其前向计算得到的损失值计算梯度，使用AllReduce操作逐个累加每个参数在所有计算设备上的梯度值，并计算累积梯度的平均值，从而确保各个计算设备上用于更新参数的梯度值是相同的。参数更新阶段，使用梯度平均值更新参数。整个计算过程如上图(b)所示。

由于在训练起始阶段，通过广播操作确保了各个计算设备上的参数一致性；反向阶段，各个计算设备上使用相同的梯度均值更新参数；因此，可以保证训练过程中各个计算设备上的参数值始终是一致的。

数据并行训练主要包括以下两种方式。

1. 各个卡的批处理大小（batch size）和单卡训练保持一致。假设单卡的批处理大小为B，数据并行训练使用的卡数为K。那么，数据并行方式下，单次迭代处理的数据量为KB。在理想情形下，数据并行训练的吞吐量是单卡训练的K倍。但实际情形下，分布式训练引入了额外的通信和计算开销，如累积各个卡上梯度值并计算平均值。这种额外的通信开销和计算开销通常较小。因此，数据并行训练相比于单卡训练的吞吐量更高，加速比通常略小于K。

2. 各个卡的批处理大小总和与单卡训练的批处理大小一致。那么，分布式训练方式下，各个卡的批处理大小为B/K。因此，分布式训练方式下，每次迭代的时间均明显小于单卡训练，从而在整体上提高训练吞吐量。


二、动态图使用方法
-----------------------

与单机单卡的普通模型训练相比，数据并行训练代码只需要补充以下四个部分：

#. 导入分布式训练需要的依赖包。
#. 初始化分布式环境。
#. 设置分布式训练需要的优化器。
#. 设置数据并行训练需要的批采样器以加载子数据集。

下面将逐一进行讲解。

2.1 导入数据并行训练需要的依赖包
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

导入必要的依赖，例如分布式训练专用的Fleet API(``paddle.distributed.fleet``)。

.. code-block::

    from paddle.distributed import fleet
    from paddle.io import DistributedBatchSampler

2.2 初始化分布式环境
^^^^^^^^^^^^^^^^^^^^^

包括定义缺省的分布式策略，然后通过将参数 ``is_collective`` 设置为True，使训练架构设定为Collective架构。

.. code-block::

    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)

2.3 设置分布式训练需要的优化器
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用Fleet API设置分布式训练模型、优化器。

.. code-block::

    model = fleet.distributed_model(model)
    optimizer = fleet.distributed_optimizer(optimizer)

2.4 设置数据并行训练需要的批采样器
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 ``DistributedBatchSampler`` 设置分布式批采样器，并传递给DataLoader，使得每个进程加载原始数据的一个子集。

.. code-block::

    dataset = RandomDataset(batch_num * batch_size)
    sampler = DistributedBatchSampler(dataset,
                                      rank=get_rank(),
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=True,)
    train_loader = DataLoader(dataset,
                              batch_sampler=sampler,
                              num_workers=1)


三、动态图运行示例
-----------------------

本节给出一个分布式任务的完整训练代码和部分训练日志。它使用了动态图数据并行策略，并通过 ``paddle.distributed.launch`` 组件启动。

3.1 完整训练代码
^^^^^^^^^^^^^^^^^^

动态图完整训练代码如下所示(train.py)：

.. code-block:: py

    # -*- coding: UTF-8 -*-
    import numpy as np
    import paddle
    # 导入必要分布式训练的依赖包
    from paddle.distributed import fleet, get_rank
    # 导入模型文件
    from paddle.vision.models import ResNet
    from paddle.vision.models.resnet import BottleneckBlock
    from paddle.io import Dataset, DistributedBatchSampler, DataLoader

    base_lr = 0.1   # 学习率
    momentum_rate = 0.9 # 冲量
    l2_decay = 1e-4 # 权重衰减

    epoch = 10  #训练迭代次数
    batch_num = 100 #每次迭代的batch数
    batch_size = 32 #训练批次大小
    class_dim = 102

    # 设置数据读取器
    class RandomDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __getitem__(self, idx):
            image = np.random.random([3, 224, 224]).astype('float32')
            label = np.random.randint(0, class_dim - 1, (1, )).astype('int64')
            return image, label

        def __len__(self):
            return self.num_samples

    # 设置优化器
    def optimizer_setting(parameter_list=None):
        optimizer = paddle.optimizer.Momentum(
            learning_rate=base_lr,
            momentum=momentum_rate,
            weight_decay=paddle.regularizer.L2Decay(l2_decay),
            parameters=parameter_list)
        return optimizer

    # 设置训练函数
    def train_model():
        # 初始化Fleet环境
        fleet.init(is_collective=True)

        model = ResNet(BottleneckBlock, 50, num_classes=class_dim)

        optimizer = optimizer_setting(parameter_list=model.parameters())
        optimizer = fleet.distributed_optimizer(optimizer)
        # 通过Fleet API获取分布式model，用于支持分布式训练
        model = fleet.distributed_model(model)

        dataset = RandomDataset(batch_num * batch_size)
        sampler = DistributedBatchSampler(dataset,
                                        rank=get_rank(),
                                        batch_size=batch_size,
                                        shuffle=False,
                                        drop_last=True,)
        train_loader = DataLoader(dataset,
                                batch_sampler=sampler,
                                num_workers=1)

        for eop in range(epoch):
            model.train()

            for batch_id, data in enumerate(train_loader()):
                img, label = data
                label.stop_gradient = True

                out = model(img)
                loss = paddle.nn.functional.cross_entropy(input=out, label=label)
                avg_loss = paddle.mean(x=loss)
                acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
                acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)

                avg_loss.backward()
                optimizer.step()
                model.clear_gradients()

                if batch_id % 5 == 0:
                    print("[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f" % (eop, batch_id, avg_loss, acc_top1, acc_top5))
    # 启动训练
    if __name__ == '__main__':
        train_model()

3.2 运行日志
^^^^^^^^^^^^^^^^^^

飞桨分布式任务可以通过 ``paddle.distributed.launch`` 组件启动。假设要运行2卡任务，只需在命令行中执行:

.. code-block::

   python -m paddle.distributed.launch --gpus=0,1 --log_dir logs train.py

您将在logs路径下看到2份日志文件，即workerlog.0和workerlog.1；日志中显示如下运行信息：

.. code-block::

    -----------  Configuration Arguments -----------
    gpus: 0,1
    heter_worker_num: None
    heter_workers:
    http_port: None
    ips: 127.0.0.1
    log_dir: logs
    ...
    ------------------------------------------------
    launch train in GPU mode!
    INFO 2022-05-19 08:07:19,137 launch_utils.py:557] Local start 2 processes. First process distributed environment info (Only For Debug): 
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                       PADDLE_TRAINER_ID                        0                      |
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:57073               |
        |                     PADDLE_TRAINERS_NUM                        2                      |
        |                PADDLE_TRAINER_ENDPOINTS         127.0.0.1:57073,127.0.0.1:11503       |
        |                     PADDLE_RANK_IN_NODE                        0                      |
        |                 PADDLE_LOCAL_DEVICE_IDS                        0                      |
        |                 PADDLE_WORLD_DEVICE_IDS                       0,1                     |
        |                     FLAGS_selected_gpus                        0                      |
        |             FLAGS_selected_accelerators                        0                      |
        +=======================================================================================+

    launch proc_id:19793 idx:0
    launch proc_id:19798 idx:1
    I0519 08:07:21.830699 19793 nccl_context.cc:82] init nccl context nranks: 2 local rank: 0 gpu id: 0 ring id: 0
    W0519 08:07:23.502454 19793 device_context.cc:525] Please NOTE: device: , GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
    W0519 08:07:23.509383 19793 device_context.cc:543] device: , cuDNN Version: 8.1.
    I0519 08:07:29.192090 19793 nccl_context.cc:114] init nccl context nranks: 2 local rank: 0 gpu id: 0 ring id: 10
    2022-05-19 08:07:29,466-INFO: [topology.py:169:__init__] HybridParallelInfo: rank_id: 0, mp_degree: 1, sharding_degree: 1, pp_degree: 1, dp_degree: 2, mp_group: [0],  sharding_group: [0], pp_group: [0], dp_group: [0, 1], check/clip group: [0]
    [Epoch 0, batch 0] loss: 6.37349, acc1: 0.00000, acc5: 0.03125
    [Epoch 0, batch 5] loss: 35.73608, acc1: 0.00000, acc5: 0.03125
    [Epoch 0, batch 10] loss: 51.47758, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 15] loss: 8.24211, acc1: 0.00000, acc5: 0.09375
    [Epoch 0, batch 20] loss: 5.11644, acc1: 0.00000, acc5: 0.00000
    [Epoch 0, batch 25] loss: 7.18224, acc1: 0.03125, acc5: 0.03125
    [Epoch 0, batch 30] loss: 5.15862, acc1: 0.00000, acc5: 0.09375
    [Epoch 0, batch 35] loss: 4.54878, acc1: 0.00000, acc5: 0.06250
    [Epoch 0, batch 40] loss: 4.61982, acc1: 0.03125, acc5: 0.12500


四、数据并行使用技巧
-----------------------

首先，我们阐述数据并行模式下学习率的设置技巧，其基本原则是学习率正比于 ``global batch size`` 。

与单卡训练相比，数据并行训练通常有两种配置：
1. 一种是保持保持所有计算设备的batch size的总和（我们称为 ``global batch size`` ）与单卡训练的batch size保持一致。这中情形下，由于数据并行训练和单卡训练的 ``global batch size`` 是一致的，通常保持数据并行模式下各个计算设备上的学习率与单卡训练一致。
2. 另一种情形是，保持数据并行模式下每个计算设备的batch size和单卡训练的batch size一致。这种情形下，数据并行模式的 ``global batch size`` 是单卡训练的 ``N` `倍。这里， ``N`` 指的是数据并行计算的设备数。因此，通常需要将数据并行模式下每个计算设备的学习率相应的设置为单卡训练的 ``N`` 倍。这样，数据并行模式下的初始学习率通常较大，不利于模型的收敛。因此，通常需要使用warm-up机制。即，在初始训练时使用较小的学习率，并逐步缓慢增加学习率，经过一定迭代次数后，学习率增长到期望的学习率。

接着，我们介绍数据集切分问题。数据并行中，我们通常将数据集切分为 ``N`` 份，每个训练卡负责训练其中的一份数据。这里， ``N`` 是数据并行的并行度。如我们前面介绍的，每一个迭代中，各个训练卡均需要做一次梯度同步。因此，我们需要确保对于每个 ``epoch`` ，各个训练卡经历相同的迭代数，否则，运行迭代数多的训练卡会一直等待通信完成。实践中，我们通常通过数据补齐或者丢弃的方式保证各个训练卡经历相同的迭代数。数据补齐的方式指的是，为某些迭代数少训练数据补充部分数据，从而保证切分后的各份数据集的迭代次数相同；丢弃的方式则是丢弃部分迭代次数较多的数据，从而保证各份数据集的迭代次数相同。

通常，在每个 ``epoch`` 需要对数据做shuffle处理。因此，根据shuffle时机的不同，有两种数据切分的方法。一种是在数据切分前做shuffle；即，首先对完整的数据做shuffle处理，做相应的数据补充或丢弃，然后做数据的切分。另一种是在数据切分后做shuffle；即，首先做数据的补充或丢弃和数据切分，然后对切分后的每一份数据分别做shuffle处理。

需要注意的是，上述只是给出一些常见的数据并行技巧。在实际使用中，用户需要根据实际业务需要，灵活处理。


五、参考文献
-----------------------

[1] `Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes <https://arxiv.org/abs/1807.11205>`_