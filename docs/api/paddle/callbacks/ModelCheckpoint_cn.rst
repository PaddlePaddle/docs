.. _cn_api_paddle_callbacks_ModelCheckpoint:

ModelCheckpoint
-------------------------------

.. py:class:: paddle.callbacks.ModelCheckpoint(save_freq=1, save_dir=None)

 ``ModelCheckpoint`` 回调类和model.fit联合使用，在训练阶段，保存模型权重和优化器状态信息。当前仅支持在固定的epoch间隔保存模型，不支持按照batch的间隔保存。

   子方法可以参考基类。

参数：
  - **save_freq** (int，可选) - 间隔多少个epoch保存模型。默认值：1。 
  - **save_dir** (int，可选) - 保存模型的文件夹。如果不设定，将不会保存模型。默认值：None。 


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

    callback = paddle.callbacks.ModelCheckpoint(save_dir='./temp')
    model.fit(train_dataset, batch_size=64, callbacks=callback)
