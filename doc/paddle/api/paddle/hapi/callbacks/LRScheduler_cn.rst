.. _cn_api_paddle_callbacks_LRScheduler:

LRScheduler
-------------------------------

.. py:class:: paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)

 ``LRScheduler`` 是一个学习率回调函数。

参数：
  - **by_step** (bool，可选) - 是否每个step都更新学习率。默认值：True。 
  - **by_epoch** (bool，可选) - 是否每个epoch都更新学习率。默认值：False。 


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

    base_lr = 1e-3
    boundaries = [5, 8]
    wamup_steps = 4
    
    def make_optimizer(parameters=None):
        momentum = 0.9
        weight_decay = 5e-4
        values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
        learning_rate = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=boundaries, values=values)
        learning_rate = paddle.optimizer.lr.LinearWarmup(
            learning_rate=learning_rate,
            warmup_steps=wamup_steps,
            start_lr=base_lr / 5.,
            end_lr=base_lr,
            verbose=True)
        optimizer = paddle.optimizer.Momentum(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            parameters=parameters)
        return optimizer
        
    optim = make_optimizer(parameters=lenet.parameters())
    model.prepare(optimizer=optim,
                loss=paddle.nn.CrossEntropyLoss(),
                metrics=paddle.metric.Accuracy())

    # if LRScheduler callback not set, an instance LRScheduler update by step 
    # will be created auto.
    model.fit(train_dataset, batch_size=64)

    # create a learning rate scheduler update by epoch
    callback = paddle.callbacks.LRScheduler(by_step=False, by_epoch=True)
    model.fit(train_dataset, batch_size=64, callbacks=callback)
    