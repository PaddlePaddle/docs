.. _cn_api_paddle_callbacks_EarlyStopping:

EarlyStopping
-------------------------------

.. py:class:: paddle.callbacks.EarlyStopping(monitor='loss', mode='auto', patience=0, verbose=1, min_delta=0, baseline=None, save_best_model=True)

在模型评估阶段，模型效果如果没有提升，``EarlyStopping`` 会让模型提前停止训练。

参数：
  - **monitor** (str，可选) - 监控量。该量作为模型是否停止学习的监控指标。默认值：'loss'。
  - **mode** (str，可选) - 可以是'auto'、'min'或者'max'。在min模式下，模型会在监控量的值不再减少时停止训练；max模式下，模型会在监控量的值不再增加时停止训练；auto模式下，实际的模式会从 ``monitor`` 推断出来。如果 ``monitor`` 中有'acc'，将会认为是max模式，其它情况下，都会被推断为min模式。默认值：'auto'。
  - **patience** (int，可选) - 多少个epoch模型效果未提升会使模型提前停止训练。默认值：0。
  - **verbose** (int，可选) - 可以是0或者1。0代表不打印模型提前停止训练的日志，1代表打印日志。默认值：1。
  - **min_delta** (int|float，可选) - 监控量最小改变值。当evaluation的监控变量改变值小于 ``min_delta`` ，就认为模型没有变化。默认值：0。
  - **baseline** (int|float，可选) - 监控量的基线。如果模型在训练 ``patience`` 个epoch后效果对比基线没有提升，将会停止训练。如果是None，代表没有基线。默认值：None。
  - **save_best_model** (bool，可选) - 是否保存效果最好的模型（监控量的值最优）。文件会保存在 ``fit`` 中传入的参数 ``save_dir`` 下，前缀名为best_model，默认值: True。

**代码示例**：

.. code-block:: python

    import paddle
    from paddle import Model
    from paddle.static import InputSpec
    from paddle.vision.models import LeNet
    from paddle.vision.datasets import MNIST
    from paddle.metric import Accuracy
    from paddle.nn import CrossEntropyLoss
    import paddle.vision.transforms as T

    device = paddle.set_device('cpu')
    sample_num = 200
    save_dir = './best_model_checkpoint'
    transform = T.Compose(
        [T.Transpose(), T.Normalize([127.5], [127.5])])
    train_dataset = MNIST(mode='train', transform=transform)
    val_dataset = MNIST(mode='test', transform=transform)

    net = LeNet()
    optim = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=net.parameters())
    inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
    labels = [InputSpec([None, 1], 'int64', 'label')]
    model = Model(net, inputs=inputs, labels=labels)

    model.prepare(
        optim,
        loss=CrossEntropyLoss(reduction="sum"),
        metrics=[Accuracy()])

    callbacks = paddle.callbacks.EarlyStopping(
        'loss',
        mode='min',
        patience=1,
        verbose=1,
        min_delta=0,
        baseline=None,
        save_best_model=True)

    model.fit(train_dataset,
                val_dataset,
                batch_size=64,
                log_freq=200,
                save_freq=10,
                save_dir=save_dir,
                epochs=20,
                callbacks=[callbacks])
