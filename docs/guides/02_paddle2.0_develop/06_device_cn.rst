.. _cn_doc_device:

单机多卡训练
==================

飞桨框架2.0增加\ ``paddle.distributed.spawn``\ 函数来启动单机多卡训练，同时原有的\ ``paddle.distributed.launch``\ 的方式依然保留。

一、launch启动
---------------------

1.1 高层API场景
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当调用\ ``paddle.Model``\高层API来实现训练时，想要启动单机多卡训练非常简单，代码不需要做任何修改，只需要在启动时增加一下参数\ ``-m paddle.distributed.launch``\ 。

.. code:: bash

    # 单机单卡启动，默认使用第0号卡
    $ python train.py

    # 单机多卡启动，默认使用当前可见的所有卡
    $ python -m paddle.distributed.launch train.py

    # 单机多卡启动，设置当前使用的第0号和第1号卡
    $ python -m paddle.distributed.launch --gpus='0,1' train.py

    # 单机多卡启动，设置当前使用第0号和第1号卡
    $ export CUDA_VISIBLE_DEVICES=0,1
    $ python -m paddle.distributed.launch train.py

1.2 基础API场景
~~~~~~~~~~~~~~~~~~

如果使用基础API实现训练，想要启动单机多卡训练，需要对单机单卡的代码进行3处修改，具体如下：

.. code:: python3

    import paddle
    # 第1处改动 导入分布式训练所需的包
    import paddle.distributed as dist

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

    # 第2处改动，初始化并行环境
    dist.init_parallel_env()

    # 用 DataLoader 实现数据加载
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 第3处改动，增加paddle.DataParallel封装
    mnist = paddle.DataParallel(mnist)
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

修改完后保存文件，然后使用跟高层API相同的启动方式即可。
**注意：** 单卡训练不支持调用\ ``init_parallel_env``\ ，请使用以下几种方式进行分布式训练。

.. code:: bash

    # 单机多卡启动，默认使用当前可见的所有卡
    $ python -m paddle.distributed.launch train.py

    # 单机多卡启动，设置当前使用的第0号和第1号卡
    $ python -m paddle.distributed.launch --gpus '0,1' train.py

    # 单机多卡启动，设置当前使用第0号和第1号卡
    $ export CUDA_VISIBLE_DEVICES=0,1
    $ python -m paddle.distributed.launch train.py

二、spawn启动
-------------------------------
launch方式启动训练，以文件为单位启动多进程，需要用户在启动时调用\ ``paddle.distributed.launch``\，对于进程的管理要求较高。飞桨框架2.0版本增加了\ ``spawn``\ 启动方式，可以更好地控制进程，在日志打印、训练退出时更友好。使用示例如下：

.. code:: python3

    from __future__ import print_function

    import paddle
    import paddle.nn as nn
    import paddle.optimizer as opt
    import paddle.distributed as dist

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear1 = nn.Linear(10, 10)
            self._linear2 = nn.Linear(10, 1)

        def forward(self, x):
            return self._linear2(self._linear1(x))

    def train(print_result=False):

        # 1. 初始化并行训练环境
        dist.init_parallel_env()

        # 2. 创建并行训练 Layer 和 Optimizer
        layer = LinearNet()
        dp_layer = paddle.DataParallel(layer)

        loss_fn = nn.MSELoss()
        adam = opt.Adam(
            learning_rate=0.001, parameters=dp_layer.parameters())

        # 3. 运行网络
        inputs = paddle.randn([10, 10], 'float32')
        outputs = dp_layer(inputs)
        labels = paddle.randn([10, 1], 'float32')
        loss = loss_fn(outputs, labels)

        if print_result is True:
            print("loss:", loss.numpy())

        loss.backward()

        adam.step()
        adam.clear_grad()

    # 使用方式1：仅传入训练函数
    # 适用场景：训练函数不需要任何参数，并且需要使用所有当前可见的GPU设备并行训练
    if __name__ == '__main__':
        dist.spawn(train)

    # 使用方式2：传入训练函数和参数
    # 适用场景：训练函数需要一些参数，并且需要使用所有当前可见的GPU设备并行训练
    if __name__ == '__main__':
        dist.spawn(train, args=(True,))

    # 使用方式3：传入训练函数、参数并指定并行进程数
    # 适用场景：训练函数需要一些参数，并且仅需要使用部分可见的GPU设备并行训练，例如：
    # 当前机器有8张GPU卡 {0,1,2,3,4,5,6,7}，此时会使用前两张卡 {0,1}；
    # 或者当前机器通过配置环境变量 CUDA_VISIBLE_DEVICES=4,5,6,7，仅使4张
    # GPU卡可见，此时会使用可见的前两张卡 {4,5}
    if __name__ == '__main__':
        dist.spawn(train, args=(True,), nprocs=2)

    # 使用方式4：传入训练函数、参数、指定进程数并指定当前使用的卡号
    # 使用场景：训练函数需要一些参数，并且仅需要使用部分可见的GPU设备并行训练，但是
    # 可能由于权限问题，无权配置当前机器的环境变量，例如：当前机器有8张GPU卡 
    # {0,1,2,3,4,5,6,7}，但你无权配置CUDA_VISIBLE_DEVICES，此时可以通过
    # 指定参数 gpus 选择希望使用的卡，例如 gpus='4,5'，
    # 可以指定使用第4号卡和第5号卡
    if __name__ == '__main__':
        dist.spawn(train, nprocs=2, gpus='4,5')
