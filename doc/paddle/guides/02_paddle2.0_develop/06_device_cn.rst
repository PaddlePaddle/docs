.. _cn_doc_device:

资源配置
==================

飞桨框架2.0增加\ ``paddle.distributed.spawn``\ 函数来启动单机多卡训练，同时原有的\ ``paddle.distributed.launch``\ 的方式依然保留。

1. 方式1、launch启动
---------------------

高层API场景
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当调用\ ``paddle.Model``\高层API来实现训练时，想要启动单机多卡训练非常简单，代码不需要做任何修改，只需要在启动时增加一下参数\ ``-m paddle.distributed.launch``\ 。

.. code:: bash

    # 单机单卡启动，默认使用第0号卡
    $ python train.py

    # 单机多卡启动，默认使用当前可见的所有卡
    $ python -m paddle.distributed.launch train.py

    # 单机多卡启动，设置当前使用的第0号和第1号卡
    $ python -m paddle.distributed.launch --selected_gpus='0,1' train.py

    # 单机多卡启动，设置当前使用第0号和第1号卡
    $ export CUDA_VISIABLE_DEVICES='0,1'
    $ python -m paddle.distributed.launch train.py

基础API场景
~~~~~~~~~~~~~~~~~~

如果使用基础API实现训练，想要启动单机多卡训练，需要对单机单卡的代码进行2处修改，具体如下：

.. code:: python3

    import paddle
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
    # 第1处改动，初始化并行环境
    dist.init_parallel_env()

    # 用 DataLoader 实现数据加载
    train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=32, shuffle=True)
    
    # 第2处改动，增加paddle.DataParallel封装
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

.. code:: bash

    # 单机单卡启动，默认使用第0号卡
    $ python train.py

    # 单机多卡启动，默认使用当前可见的所有卡
    $ python -m paddle.distributed.launch train.py

    # 单机多卡启动，设置当前使用的第0号和第1号卡
    $ python -m paddle.distributed.launch --selected_gpus '0,1' train.py

    # 单机多卡启动，设置当前使用第0号和第1号卡
    $ export CUDA_VISIABLE_DEVICES='0,1'
    $ python -m paddle.distributed.launch train.py

2. 方式2、spawn启动
-------------------------------
launch方式启动训练，以文件为单位启动多进程，需要用户在启动时调用\ ``paddle.distributed.launch``\，对于进程的管理要求较高。飞桨框架2.0版本增加了\ ``spawn``\ 启动方式，可以更好地控制进程，在日志打印、训练退出时更友好。

.. code:: python3

    # 启动train多进程训练，默认使用所有可见的GPU卡
    if __name__ == '__main__':
        dist.spawn(train)

    # 启动train函数2个进程训练，默认使用当前可见的前2张卡
    if __name__ == '__main__':
        dist.spawn(train, nprocs=2)

    # 启动train函数2个进程训练，默认使用第4号和第5号卡
    if __name__ == '__main__':
        dist.spawn(train, nprocs=2, selelcted_gpus='4,5')

