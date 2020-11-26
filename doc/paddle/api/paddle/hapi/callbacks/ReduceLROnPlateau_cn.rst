.. _cn_api_paddle_callbacks_ReduceLROnPlateau:

ReduceLROnPlateau
-------------------------------

.. py:class:: paddle.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0)

 当评估指标停止改善时，降低学习率。模型通常会因学习率降低2-10倍而受益。因此监视一个评价指标，如果这个指标在几个epoch内没有改善，就降低学习率。

参数：
  - **monitor** (str，可选) - 监视的指标名称。默认值：'loss'。 
  - **factor** (float，可选) - 学习率减小的因子。 `new_lr = lr * factor` 。默认值：0.1。 
  - **patience** (int，可选) - 多少个epoch监视的指标没有提升后就减小学习率。 默认值：10。 
  - **verbose** (int，可选) - 可视化的模式。0表示不打印任何信息，1表示打印信息。默认值：1。 
  - **mode** (int，可选) - 必须是 `{'auto', 'min', 'max'}` 中的值。`'min'` 表示学习率
      会减少当监视的指标不再下降。 `'max'` 表示学习率会减少当监视的指标不再上升。 `'auto'` 
      会根据监视指标的名字来推理是使用min还是max模式，如果名字中包含acc则使用max模式，
      否则使用min模式。 默认值：'auto'。 
  - **min_delta** (float，可选) - 评判指标增大或减小的阈值。默认值：0。 
  - **cooldown** (int，可选) - 学习率减少后至少经过多少个epoch在进行正常的减少策略。默认值：0。 
  - **min_lr** (int，可选) - 学习率减小后的下限。默认值：0。 


**代码示例**：

.. code-block:: python

    import paddle
    from paddle import Model
    from paddle.static import InputSpec
    from paddle.vision.models import LeNet
    from paddle.vision.datasets import MNIST
    from paddle.metric import Accuracy
    from paddle.nn.layer.loss import CrossEntropyLoss
    import paddle.vision.transforms as T  
    sample_num = 200
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
        loss=CrossEntropyLoss(),
        metrics=[Accuracy()])  
    callbacks = paddle.callbacks.ReduceLROnPlateau(patience=3, verbose=1)
    model.fit(train_dataset,
                val_dataset,
                batch_size=64,
                log_freq=200,
                save_freq=10,
                epochs=20,
                callbacks=[callbacks])