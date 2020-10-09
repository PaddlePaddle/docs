.. _cn_api_paddle_callbacks_ProgBarLogger:

ProgBarLogger
-------------------------------

.. py:class:: paddle.callbacks.ProgBarLogger(log_freq=1, verbose=2)

 ``ProgBarLogger`` 是一个日志回调类。

参数：
  - **log_freq** (int，可选) - 损失值和指标打印的频率。默认值：1。
  - **verbose** (int，可选) - 打印信息的模式。设置为0时，不打印信息；设置为1时，使用进度条的形式打印信息；是指为2时，使用行的形式打印信息。默认值：2。


**代码示例**：

.. code-block:: python

    import paddle
    from paddle.static import InputSpec

    inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
    labels = [InputSpec([None, 1], 'int64', 'label')]

    train_dataset = paddle.vision.datasets.MNIST(mode='train')

    model = paddle.Model(paddle.vision.LeNet(classifier_activation=None),
        inputs, labels)

    optim = paddle.optimizer.Adam(0.001)
    model.prepare(optimizer=optim,
                loss=paddle.nn.CrossEntropyLoss(),
                metrics=paddle.metric.Accuracy())

    callback = paddle.callbacks.ProgBarLogger(log_freq=10)
    model.fit(train_dataset, batch_size=64, callbacks=callback)