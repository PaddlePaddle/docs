.. _cn_api_paddle_callbacks_VisualDL:

VisualDL
-------------------------------

.. py:class:: paddle.callbacks.VisualDL(log_dir)

 ``VisualDL`` 是一个visualdl的回调类。

参数：
  - **log_dir** (str) - 输出日志保存的路径。  


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
    eval_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

    net = paddle.vision.LeNet()
    model = paddle.Model(net, inputs, labels)

    optim = paddle.optimizer.Adam(0.001, parameters=net.parameters())
    model.prepare(optimizer=optim,
                loss=paddle.nn.CrossEntropyLoss(),
                metrics=paddle.metric.Accuracy())
    
    ## uncomment following lines to fit model with visualdl callback function
    # callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')
    # model.fit(train_dataset, eval_dataset, batch_size=64, callbacks=callback)