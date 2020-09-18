.. _cn_api_paddle_Model:

Model
-------------------------------

.. py:class:: paddle.Model()

 ``Model`` 对象是一个具备训练、测试、推理的神经网络。该对象同时支持静态图和动态图模式，通过 ``paddle.disable_static()`` 来切换。需要注意的是，该开关需要在实例化 ``Model`` 对象之前使用。输入需要使用 ``paddle.static.InputSpec`` 来定义。

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec

    device = paddle.set_device('cpu') # or 'gpu'
    # if use static graph, do not set
    paddle.disable_static(device)

    net = nn.Sequential(
        nn.Linear(784, 200),
        nn.Tanh(),
        nn.Linear(200, 10))

    input = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')
    
    model = paddle.Model(net, input, label)
    optim = paddle.optimizer.SGD(learning_rate=1e-3,
        parameters=model.parameters())
    model.prepare(optim,
                    paddle.nn.CrossEntropyLoss(),
                    paddle.metric.Accuracy())
    
    data = paddle.vision.datasets.MNIST(mode='train', chw_format=False)
    model.fit(data, epochs=2, batch_size=32, verbose=1)


.. py:function:: train_batch(inputs, labels=None)

在一个批次的数据上进行训练。

参数：
    - **inputs** (list) - 1维列表，每个元素都是一批次的输入数据，数据类型为 ``numpy.ndarray``。
    - **labels** (list) - 1维列表，每个元素都是一批次的输入标签，数据类型为 ``numpy.ndarray`` 。默认值：None。
    
返回：如果没有定义评估函数，则返回包含了训练损失函数的值的列表；如果定义了评估函数，则返回一个元组（损失函数的列表，评估指标的列表）。


**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec

    device = paddle.set_device('cpu') # or 'gpu'
    paddle.disable_static(device)

    net = nn.Sequential(
        nn.Linear(784, 200),
        nn.Tanh(),
        nn.Linear(200, 10))

    input = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')
    model = paddle.Model(net, input, label)
    optim = paddle.optimizer.SGD(learning_rate=1e-3,
        parameters=model.parameters())
    model.prepare(optim, paddle.nn.CrossEntropyLoss())
    data = np.random.random(size=(4,784)).astype(np.float32)
    label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)
    loss = model.train_batch([data], [label])
    print(loss)

.. py:function:: eval_batch(inputs, labels=None)

在一个批次的数据上进行评估。

参数：
    - **inputs** (list) - 1维列表，每个元素都是一批次的输入数据，数据类型为 ``numpy.ndarray`` 。
    - **labels** (list) - 1维列表，每个元素都是一批次的输入标签，数据类型为 ``numpy.ndarray`` 。默认值：None。
    
返回：如果没有定义评估函数，则返回包含了预测损失函数的值的列表；如果定义了评估函数，则返回一个元组（损失函数的列表，评估指标的列表）。

返回类型：list

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec

    device = paddle.set_device('cpu') # or 'gpu'
    paddle.disable_static(device)

    net = nn.Sequential(
        nn.Linear(784, 200),
        nn.Tanh(),
        nn.Linear(200, 10))

    input = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')
    model = paddle.Model(net, input, label)
    optim = paddle.optimizer.SGD(learning_rate=1e-3,
        parameters=model.parameters())
    model.prepare(optim,
                paddle.nn.CrossEntropyLoss())
    data = np.random.random(size=(4,784)).astype(np.float32)
    label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)
    loss = model.eval_batch([data], [label])
    print(loss)

.. py:function:: test_batch(inputs)

在一个批次的数据上进行测试。

参数：
    - **inputs** (list) - 1维列表，每个元素都是一批次的输入数据，数据类型为 ``numpy.ndarray`` 。
    
返回：一个列表，包含了模型的输出。

返回类型：list

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec

    device = paddle.set_device('cpu') # or 'gpu'
    paddle.disable_static(device)

    net = nn.Sequential(
        nn.Linear(784, 200),
        nn.Tanh(),
        nn.Linear(200, 10),
        nn.Softmax())

    input = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')
    model = paddle.Model(net, input, label)
    model.prepare()
    data = np.random.random(size=(4,784)).astype(np.float32)
    out = model.test_batch([data])
    print(out)

.. py:function:: save(path, training=True):

将模型的参数和训练过程中优化器的信息保存到指定的路径，以及推理所需的参数与文件。如果training=True，所有的模型参数都会保存到一个后缀为 ``.pdparams`` 的文件中。
所有的优化器信息和相关参数，比如 ``Adam`` 优化器中的 ``beta1`` ， ``beta2`` ，``momentum`` 等，都会被保存到后缀为 ``.pdopt``。如果优化器比如SGD没有参数，则该不会产生该文件。如果training=False，则不会保存上述说的文件。只会保存推理需要的参数文件和模型文件。

参数：
    - **path** (str) - 保存的文件名前缀。格式如 ``dirname/file_prefix`` 或者 ``file_prefix`` 。
    - **training** (bool，可选) - 是否保存训练的状态，包括模型参数和优化器参数等。如果为False，则只保存推理所需的参数与文件。默认值：True。
    
返回：None

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec

    class Mnist(nn.Layer):
        def __init__(self):
            super(Mnist, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 200),
                nn.Tanh(),
                nn.Linear(200, 10),
                nn.Softmax())

        def forward(self, x):
            return self.net(x)

    dynamic = True  # False
    device = paddle.set_device('cpu')
    # if use static graph, do not set
    paddle.disable_static(device) if dynamic else None

    input = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')
    model = paddle.Model(Mnist(), input, label)
    optim = paddle.optimizer.SGD(learning_rate=1e-3,
        parameters=model.parameters())
    model.prepare(optim, paddle.nn.CrossEntropyLoss())
    data = paddle.vision.datasets.MNIST(mode='train', chw_format=False)
    model.fit(data, epochs=1, batch_size=32, verbose=0)
    model.save('checkpoint/test')  # save for training
    model.save('inference_model', False)  # save for inference

.. py:function:: load(path, skip_mismatch=False, reset_optimizer=False):

从指定的文件中载入模型参数和优化器参数，如果不想恢复优化器参数信息，优化器信息文件可以不存在。需要注意的是：参数名称的检索是根据保存模型时结构化的名字，当想要载入参数进行迁移学习时要保证预训练模型和当前的模型的参数有一样结构化的名字。

参数：
    - **path** (str) - 保存参数或优化器信息的文件前缀。格式如 ``path.pdparams`` 或者 ``path.pdopt`` ，后者是非必要的，如果不想恢复优化器信息。
    - **skip_mismatch** (bool) - 是否需要跳过保存的模型文件中形状或名称不匹配的参数，设置为 ``False`` 时，当遇到不匹配的参数会抛出一个错误。默认值：False。
    - **reset_optimizer** (bool) - 设置为 ``True`` 时，会忽略提供的优化器信息文件。否则会载入提供的优化器信息。默认值：False。
    
返回：None

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec
    
    device = paddle.set_device('cpu')
    paddle.disable_static(device)

    input = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')
    model = paddle.Model(nn.Sequential(
        nn.Linear(784, 200),
        nn.Tanh(),
        nn.Linear(200, 10),
        nn.Softmax()),
        input,
        label)
    model.save('checkpoint/test')
    model.load('checkpoint/test')

.. py:function:: parameters(*args, **kwargs):

返回一个包含模型所有参数的列表。
    
返回：在静态图中返回一个包含 ``Parameter`` 的列表，在动态图中返回一个包含 ``ParamBase`` 的列表。

**代码示例**：

.. code-block:: python
    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec

    paddle.disable_static()

    input = InputSpec([None, 784], 'float32', 'x')
    label = InputSpec([None, 1], 'int64', 'label')
    model = paddle.Model(nn.Sequential(
        nn.Linear(784, 200),
        nn.Tanh(),
        nn.Linear(200, 10)),
        input,
        label)
    params = model.parameters()


.. py:function:: prepare(optimizer=None, loss_function=None, metrics=None):

配置模型所需的部件，比如优化器、损失函数和评价指标。

参数：
    - **optimizer** (Optimizer) - 当训练模型的，该参数必须被设定。当评估或测试的时候，该参数可以不设定。默认值：None。
    - **loss_function** (Loss) - 当训练模型的，该参数必须被设定。默认值：None。
    - **metrics** (Metric|list[Metric]) - 当该参数被设定时，所有给定的评估方法会在训练和测试时被运行，并返回对应的指标。默认值：None。


.. py:function:: fit(train_data=None, eval_data=None, batch_size=1, epochs=1, eval_freq=1, log_freq=10, save_dir=None, save_freq=1, verbose=2, drop_last=False, shuffle=True, num_workers=0, callbacks=None):

训练模型。当 ``eval_data`` 给定时，会在 ``eval_freq`` 个 ``epoch`` 后进行一次评估。

参数：
    - **train_data** (Dataset|DataLoader) - 一个可迭代的数据源，推荐给定一个 ``paddle paddle.io.Dataset`` 或 ``paddle.io.Dataloader`` 的实例。默认值：None。
    - **eval_data** (Dataset|DataLoader) - 一个可迭代的数据源，推荐给定一个 ``paddle paddle.io.Dataset`` 或 ``paddle.io.Dataloader`` 的实例。当给定时，会在每个 ``epoch`` 后都会进行评估。默认值：None。
    - **batch_size** (int) - 训练数据或评估数据的批大小，当 ``train_data`` 或 ``eval_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：1。
    - **epochs** (int) - 训练的轮数。默认值：1。
    - **eval_freq** (int) - 评估的频率，多少个 ``epoch`` 评估一次。默认值：1。
    - **log_freq** (int) - 日志打印的频率，多少个 ``step`` 打印一次日志。默认值：1。
    - **save_dir** (str|None) - 保存模型的文件夹，如果不设定，将不保存模型。默认值：None。
    - **save_freq** (int) - 保存模型的频率，多少个 ``epoch`` 保存一次模型。默认值：1。
    - **verbose** (int) - 可视化的模型，必须为0，1，2。当设定为0时，不打印日志，设定为1时，使用进度条的方式打印日志，设定为2时，一行一行地打印日志。默认值：2。
    - **drop_last** (bool) - 是否丢弃训练数据中最后几个不足设定的批次大小的数据。默认值：False。
    - **shuffle** (bool) - 是否对训练数据进行洗牌。当 ``train_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：True。
    - **num_workers** (int) - 启动子进程用于读取数据的数量。当 ``train_data`` 和 ``eval_data`` 都为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：True。
    - **callbacks** (Callback|list[Callback]|None) -  ``Callback`` 的一个实例或实例列表。该参数不给定时，默认会插入 ``ProgBarLogger`` 和 ``ModelCheckpoint`` 这两个实例。默认值：None。

返回：None

**代码示例**：

.. code-block:: python

    # 1. 使用Dataset训练，并设置batch_size的例子。
    import paddle
    from paddle.static import InputSpec

    dynamic = True
    device = paddle.set_device('cpu') # or 'gpu'
    paddle.disable_static(device) if dynamic else None

    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    val_dataset = paddle.vision.datasets.MNIST(mode='test')

    input = InputSpec([None, 1, 28, 28], 'float32', 'image')
    label = InputSpec([None, 1], 'int64', 'label')

    model = paddle.Model(
        paddle.vision.models.LeNet(classifier_activation=None),
        input, label)
    optim = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=model.parameters())
    model.prepare(
        optim,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 2)))
    model.fit(train_dataset,
            val_dataset,
            epochs=2,
            batch_size=64,
            save_dir='mnist_checkpoint')

    # 2. 使用Dataloader训练的例子.

    import paddle
    from paddle.static import InputSpec

    dynamic = True
    device = paddle.set_device('cpu') # or 'gpu'
    paddle.disable_static(device) if dynamic else None

    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    train_loader = paddle.io.DataLoader(train_dataset,
        places=device, batch_size=64)
    val_dataset = paddle.vision.datasets.MNIST(mode='test')
    val_loader = paddle.io.DataLoader(val_dataset,
        places=device, batch_size=64)

    input = InputSpec([None, 1, 28, 28], 'float32', 'image')
    label = InputSpec([None, 1], 'int64', 'label')

    model = paddle.Model(
        paddle.vision.models.LeNet(classifier_activation=None), input, label)
    optim = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=model.parameters())
    model.prepare(
        optim,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 2)))
    model.fit(train_loader,
            val_loader,
            epochs=2,
            save_dir='mnist_checkpoint')


.. py:function:: evaluate(eval_data, batch_size=1, log_freq=10, verbose=2, num_workers=0, callbacks=None):

在输入数据上，评估模型的损失函数值和评估指标。

参数：
    - **eval_data** (Dataset|DataLoader) - 一个可迭代的数据源，推荐给定一个 ``paddle paddle.io.Dataset`` 或 ``paddle.io.Dataloader`` 的实例。默认值：None。
    - **batch_size** (int) - 训练数据或评估数据的批大小，当 ``eval_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：1。
    - **log_freq** (int) - 日志打印的频率，多少个 ``step`` 打印一次日志。默认值：1。
    - **verbose** (int) - 可视化的模型，必须为0，1，2。当设定为0时，不打印日志，设定为1时，使用进度条的方式打印日志，设定为2时，一行一行地打印日志。默认值：2。
    - **num_workers** (int) - 启动子进程用于读取数据的数量。当 ``eval_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：True。
    - **callbacks** (Callback|list[Callback]|None) -  ``Callback`` 的一个实例或实例列表。该参数不给定时，默认会插入 ``ProgBarLogger`` 和 ``ModelCheckpoint`` 这两个实例。默认值：None。

返回：None

**代码示例**：

.. code-block:: python

    # declarative mode
    import paddle
    from paddle.static import InputSpec

    # declarative mode
    val_dataset = paddle.vision.datasets.MNIST(mode='test')

    input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
    label = InputSpec([None, 1], 'int64', 'label')
    model = paddle.Model(paddle.vision.models.LeNet(), input, label)
    model.prepare(metrics=paddle.metric.Accuracy())
    result = model.evaluate(val_dataset, batch_size=64)
    print(result)

    # imperative mode
    paddle.disable_static()
    model = paddle.Model(paddle.vision.models.LeNet(), input, label)
    model.prepare(metrics=paddle.metric.Accuracy())
    result = model.evaluate(val_dataset, batch_size=64)
    print(result)


.. py:function:: predict(test_data, batch_size=1, num_workers=0, stack_outputs=False, callbacks=None):

在输入数据上，预测模型的输出。

参数：
    - **test_data** (Dataset|DataLoader) - 一个可迭代的数据源，推荐给定一个 ``paddle paddle.io.Dataset`` 或 ``paddle.io.Dataloader`` 的实例。默认值：None。
    - **batch_size** (int) - 训练数据或评估数据的批大小，当 ``eval_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：1。
    - **num_workers** (int) - 启动子进程用于读取数据的数量。当 ``eval_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：True。
    - **stack_outputs** (bool) - 是否将输出进行堆叠。默认值：False。
    - **callbacks** (Callback|list[Callback]|None) -  ``Callback`` 的一个实例或实例列表。默认值：None。

返回：None

**代码示例**：

.. code-block:: python

    # declarative mode
    import numpy as np
    import paddle
    from paddle.static import InputSpec

    class MnistDataset(paddle.vision.datasets.MNIST):
        def __init__(self, mode, return_label=True):
            super(MnistDataset, self).__init__(mode=mode)
            self.return_label = return_label

        def __getitem__(self, idx):
            img = np.reshape(self.images[idx], [1, 28, 28])
            if self.return_label:
                return img, np.array(self.labels[idx]).astype('int64')
            return img,

        def __len__(self):
            return len(self.images)

    test_dataset = MnistDataset(mode='test', return_label=False)

    # declarative mode
    input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
    model = paddle.Model(paddle.vision.models.LeNet(), input)
    model.prepare()

    result = model.predict(test_dataset, batch_size=64)
    print(len(result[0]), result[0][0].shape)

    # imperative mode
    device = paddle.set_device('cpu')
    paddle.disable_static(device)
    model = paddle.Model(paddle.vision.models.LeNet(), input)
    model.prepare()
    result = model.predict(test_dataset, batch_size=64)
    print(len(result[0]), result[0][0].shape)


.. py:function:: summary(input_size=None, batch_size=None, dtype=None):

打印网络的基础结构和参数信息。

参数：
    - **input_size** (tuple|InputSpec|list[tuple|InputSpec，可选) - 输入张量的大小。如果网络只有一个输入，那么该值需要设定为tuple或InputSpec。如果模型有多个输入。那么该值需要设定为list[tuple|InputSpec]，包含每个输入的shape。如果该值没有设置，会将 ``self._inputs`` 作为输入。默认值：None。
    - **batch_size** (int，可选) - 输入张量的批大小。默认值：None。
    - **dtypes** (str，可选) - 输入张量的数据类型，如果没有给定，默认使用 ``float32`` 类型。默认值：None。

返回：字典：包含网络全部参数的大小和全部可训练参数的大小。

**代码示例**：

.. code-block:: python

    import paddle
    from paddle.static import InputSpec

    dynamic = True
    device = paddle.set_device('cpu')
    paddle.disable_static(device) if dynamic else None

    input = InputSpec([None, 1, 28, 28], 'float32', 'image')
    label = InputSpec([None, 1], 'int64', 'label')

    model = paddle.Model(paddle.vision.LeNet(classifier_activation=None),
        input, label)
    optim = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=model.parameters())
    model.prepare(
        optim,
        paddle.nn.CrossEntropyLoss())

    params_info = model.summary()
    print(params_info)
