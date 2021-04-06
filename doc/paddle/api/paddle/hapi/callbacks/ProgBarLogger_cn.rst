.. _cn_api_paddle_callbacks_ProgBarLogger:

ProgBarLogger
-------------------------------

.. py:class:: paddle.callbacks.ProgBarLogger(log_freq=1, verbose=2)

 ``ProgBarLogger`` 是一个日志回调类，用来打印损失函数和评估指标。支持静默模式、进度条模式、每次打印一行三种模式，详细的参考下面参数注释。

参数：
  - **log_freq** (int，可选) - 损失值和指标打印的频率。默认值：1。
  - **verbose** (int，可选) - 打印信息的模式。设置为0时，不打印信息；
    设置为1时，使用进度条的形式打印信息；设置为2时，使用行的形式打印信息。
    设置为3时，会在2的基础上打印详细的计时信息，比如 ``average_reader_cost``。
    默认值：2。


**代码示例**：

.. code-block:: python

    import paddle
    import paddle.vision.transforms as T
    from paddle.static import InputSpec

    inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
    labels = [InputSpec([None, 1], 'int64', 'label')]

    transform = T.Compose([
        T.Transpose(),
        T.Normalize([127.5], [127.5])
    ])
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)

    lenet = paddle.vision.LeNet()
    model = paddle.Model(lenet,
        inputs, labels)

    optim = paddle.optimizer.Adam(0.001, parameters=lenet.parameters())
    model.prepare(optimizer=optim,
                loss=paddle.nn.CrossEntropyLoss(),
                metrics=paddle.metric.Accuracy())

    callback = paddle.callbacks.ProgBarLogger(log_freq=10)
    model.fit(train_dataset, batch_size=64, callbacks=callback)


    import paddle
    import paddle.vision.transforms as T
    from paddle.vision.datasets import MNIST
    from paddle.static import InputSpec

    inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
    labels = [InputSpec([None, 1], 'int64', 'label')]

    transform = T.Compose([
        T.Transpose(),
        T.Normalize([127.5], [127.5])
    ])
    train_dataset = MNIST(mode='train', transform=transform)

    lenet = paddle.vision.LeNet()
    model = paddle.Model(lenet,
        inputs, labels)

    optim = paddle.optimizer.Adam(0.001, parameters=lenet.parameters())
    model.prepare(optimizer=optim,
                loss=paddle.nn.CrossEntropyLoss(),
                metrics=paddle.metric.Accuracy())

    callback = paddle.callbacks.ProgBarLogger(log_freq=10)
    model.fit(train_dataset, batch_size=64, callbacks=callback)
