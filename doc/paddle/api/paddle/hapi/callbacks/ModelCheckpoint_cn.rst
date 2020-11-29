.. _cn_api_paddle_callbacks_ModelCheckpoint:

ModelCheckpoint
-------------------------------

.. py:class:: paddle.callbacks.ModelCheckpoint(save_freq=1, save_dir=None)

 ``ModelCheckpoint`` 是一个日志回调类。

参数：
  - **save_freq** (int，可选) - 间隔多少个epoch保存模型。默认值：1。 
  - **save_dir** (int，可选) - 保存模型的文件夹。如果不设定，将不会保存模型。默认值：None。 


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

    callback = paddle.callbacks.ModelCheckpoint(save_dir='./temp')
    model.fit(train_dataset, batch_size=64, callbacks=callback)