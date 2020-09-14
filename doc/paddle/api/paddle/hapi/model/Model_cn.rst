.. _cn_api_paddle_incubate_hapi_model_Model:

Model
-------------------------------

.. py:class:: paddle.incubate.hapi.model.Model()

 ``Model`` 对象是一个具备训练、测试、推理的神经网络。该对象同时支持静态图和动态图模式，通过 ``fluid.enable_dygraph()`` 来切换。需要注意的是，该开关需要在实例化 ``Model`` 对象之前使用。 在静态图模式下，输入需要使用 ``hapi.Input`` 来定义。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.fluid as fluid

    from paddle.incubate.hapi.model import Model, Input, set_device
    from paddle.incubate.hapi.loss import CrossEntropy
    from paddle.incubate.hapi.datasets import MNIST
    from paddle.incubate.hapi.metrics import Accuracy

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self._fc = fluid.dygraph.Linear(784, 10, act='softmax')
        def forward(self, x):
            y = self._fc(x)
            return y
    device = set_device('cpu')

    # 切换成动态图模式，默认使用静态图模式
    fluid.enable_dygraph(device)

    model = MyModel()
    optim = fluid.optimizer.SGD(learning_rate=1e-3,
        parameter_list=model.parameters())

    inputs = [Input([None, 784], 'float32', name='x')]
    labels = [Input([None, 1], 'int64', name='label')]

    mnist_data = MNIST(mode='train', chw_format=False)
    model.prepare(optim,
                    CrossEntropy(average=True),
                    Accuracy(),
                    inputs,
                    labels,
                    device=device)
    model.fit(mnist_data, epochs=2, batch_size=32, verbose=1)


.. py:function:: train_batch(inputs, labels=None)

在一个批次的数据上进行训练。

参数：
    - **inputs** (list) - 1维列表，每个元素都是一批次的输入数据，数据类型为 ``numpy.ndarray``。
    - **labels** (list) - 1维列表，每个元素都是一批次的输入标签，数据类型为 ``numpy.ndarray`` 。默认值：None。
    
返回：一个列表，包含了训练损失函数的值，如果定义了评估函数，还会包含评估函数得到的指标。

返回类型：list

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid

    from paddle.fluid.dygraph import Linear
    from paddle.incubate.hapi.loss import CrossEntropy
    from paddle.incubate.hapi.model import Model, Input, set_device

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self._fc = Linear(784, 10, act='softmax')
        def forward(self, x):
            y = self._fc(x)
            return y

    device = set_device('cpu')
    fluid.enable_dygraph(device)

    model = MyModel()
    optim = fluid.optimizer.SGD(learning_rate=1e-3,
        parameter_list=model.parameters())

    inputs = [Input([None, 784], 'float32', name='x')]
    labels = [Input([None, 1], 'int64', name='label')]
    model.prepare(optim,
                CrossEntropy(average=True),
                inputs=inputs,
                labels=labels,
                device=device)
    data = np.random.random(size=(4,784)).astype(np.float32)
    label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)
    loss = model.train_batch([data], [label])
    print(loss)

.. py:function:: eval_batch(inputs, labels=None)

在一个批次的数据上进行评估。

参数：
    - **inputs** (list) - 1维列表，每个元素都是一批次的输入数据，数据类型为 ``numpy.ndarray`` 。
    - **labels** (list) - 1维列表，每个元素都是一批次的输入标签，数据类型为 ``numpy.ndarray`` 。默认值：None。
    
返回：一个列表，包含了评估损失函数的值，如果定义了评估函数，还会包含评估函数得到的指标。

返回类型：list

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid

    from paddle.incubate.hapi.loss import CrossEntropy
    from paddle.incubate.hapi.model import Model, Input, set_device

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self._fc = fluid.dygraph.Linear(784, 10, act='softmax')
        def forward(self, x):
            y = self._fc(x)
            return y

    device = set_device('cpu')
    fluid.enable_dygraph(device)

    model = MyModel()
    optim = fluid.optimizer.SGD(learning_rate=1e-3,
        parameter_list=model.parameters())

    inputs = [Input([None, 784], 'float32', name='x')]
    labels = [Input([None, 1], 'int64', name='label')]
    model.prepare(optim,
                CrossEntropy(average=True),
                inputs=inputs,
                labels=labels,
                device=device)
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
    import paddle.fluid as fluid
    from paddle.incubate.hapi.model import Model, Input, set_device

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self._fc = fluid.dygraph.Linear(784, 1, act='softmax')
        def forward(self, x):
            y = self._fc(x)
            return y

    device = set_device('cpu')
    fluid.enable_dygraph(device)

    model = MyModel()
    inputs = [Input([None, 784], 'float32', name='x')]
    model.prepare(inputs=inputs,
                device=device)
    data = np.random.random(size=(4,784)).astype(np.float32)
    out = model.eval_batch([data])
    print(out)

.. py:function:: save(path):

将模型的参数和训练过程中优化器的信息保存到指定的路径。所有的模型参数都会保存到一个后缀为 ``.pdparams`` 的文件中。
所有的优化器信息和相关参数，比如 ``Adam`` 优化器中的 ``beta1`` ， ``beta2`` ，``momentum`` 等，都会被保存到后缀为 ``.pdopt`` 
的文件中。

参数：
    - **path** (str) - 保存的文件名前缀。格式如 ``dirname/file_prefix`` 或者 ``file_prefix`` 。
    
返回：None

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.incubate.hapi.model import Model, set_device
    
    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self._fc = fluid.dygraph.Linear(784, 1, act='softmax')
        def forward(self, x):
            y = self._fc(x)
            return y
    
    device = set_device('cpu')
    fluid.enable_dygraph(device)
    model = MyModel()
    model.save('checkpoint/test')

.. py:function:: load(path, skip_mismatch=False, reset_optimizer=False):

从指定的文件中载入模型参数和优化器参数，如果不想恢复优化器参数信息，优化器信息文件可以不存在。

参数：
    - **path** (str) - 保存参数或优化器信息的文件前缀。格式如 ``path.pdparams`` 或者 ``path.pdopt`` ，后者是非必要的，如果不想恢复优化器信息。
    - **skip_mismatch** (bool) - 是否需要跳过保存的模型文件中形状或名称不匹配的参数，设置为 ``False`` 时，当遇到不匹配的参数会抛出一个错误。默认值：False。
    - **reset_optimizer** (bool) - 设置为 ``True`` 时，会忽略提供的优化器信息文件。否则会载入提供的优化器信息。默认值：False。
    
返回：None

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.incubate.hapi.model import Model, set_device
    
    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self._fc = fluid.dygraph.Linear(784, 1, act='softmax')
        def forward(self, x):
            y = self._fc(x)
            return y
    
    device = set_device('cpu')
    fluid.enable_dygraph(device)
    model = MyModel()
    model.load('checkpoint/test')

.. py:function:: parameters(*args, **kwargs):

返回一个包含模型所有参数的列表。
    
返回：在静态图中返回一个包含 ``Parameter`` 的列表，在动态图中返回一个包含 ``ParamBase`` 的列表。

**代码示例**：

.. code-block:: python
    import paddle.fluid as fluid

    from paddle.incubate.hapi.model import Model, Input, set_device

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self._fc = fluid.dygraph.Linear(20, 10, act='softmax')
        def forward(self, x):
            y = self._fc(x)
            return y

    fluid.enable_dygraph()
    model = MyModel()
    params = model.parameters()


.. py:function:: prepare(optimizer=None, loss_function=None, metrics=None, inputs=None, labels=None, device=None):

返回一个包含模型所有参数的列表。

参数：
    - **optimizer** (Optimizer) - 当训练模型的，该参数必须被设定。当评估或测试的时候，该参数可以不设定。默认值：None。
    - **loss_function** (Loss) - 当训练模型的，该参数必须被设定。默认值：None。
    - **metrics** (Metric|list[Metric]) - 当该参数被设定时，所有给定的评估方法会在训练和测试时被运行，并返回对应的指标。默认值：None。
    - **inputs** (Input|list[Input]|dict) - 网络的输入，对于静态图，该参数必须给定。默认值：None。
    - **labels** (Input|list[Input]|dict) - 标签，网络的输入。对于静态图，在训练和评估时该参数必须给定。默认值：None。
    - **device** (str|fluid.CUDAPlace|fluid.CPUPlace|None) - 网络运行的设备，当不指定时，会根据环境和安装的 ``paddle`` 自动选择。默认值：None。

返回：None

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
    import paddle.fluid as fluid

    from paddle.incubate.hapi.model import Model, Input, set_device
    from paddle.incubate.hapi.loss import CrossEntropy
    from paddle.incubate.hapi.metrics import Accuracy
    from paddle.incubate.hapi.datasets import MNIST
    from paddle.incubate.hapi.vision.models import LeNet

    dynamic = True
    device = set_device('cpu')
    fluid.enable_dygraph(device) if dynamic else None

    train_dataset = MNIST(mode='train')
    val_dataset = MNIST(mode='test')

    inputs = [Input([None, 1, 28, 28], 'float32', name='image')]
    labels = [Input([None, 1], 'int64', name='label')]

    model = LeNet()
    optim = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=model.parameters())
    model.prepare(
        optim,
        CrossEntropy(),
        Accuracy(topk=(1, 2)),
        inputs=inputs,
        labels=labels,
        device=device)
    model.fit(train_dataset,
            val_dataset,
            epochs=2,
            batch_size=64,
            save_dir='mnist_checkpoint')

    # 2. 使用Dataloader训练的例子.

    from paddle.incubate.hapi.model import Model, Input, set_device
    from paddle.incubate.hapi.loss import CrossEntropy
    from paddle.incubate.hapi.metrics import Accuracy
    from paddle.incubate.hapi.datasets import MNIST
    from paddle.incubate.hapi.vision.models import LeNet

    dynamic = True
    device = set_device('cpu')
    fluid.enable_dygraph(device) if dynamic else None

    train_dataset = MNIST(mode='train')
    train_loader = fluid.io.DataLoader(train_dataset,
        places=device, batch_size=64)
    val_dataset = MNIST(mode='test')
    val_loader = fluid.io.DataLoader(val_dataset,
        places=device, batch_size=64)

    inputs = [Input([None, 1, 28, 28], 'float32', name='image')]
    labels = [Input([None, 1], 'int64', name='label')]

    model = LeNet()
    optim = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=model.parameters())
    model.prepare(
        optim,
        CrossEntropy(),
        Accuracy(topk=(1, 2)),
        inputs=inputs,
        labels=labels,
        device=device)
    model.fit(train_loader,
            val_loader,
            epochs=2,
            save_dir='mnist_checkpoint')


.. py:function:: evaluate(eval_data, batch_size=1, log_freq=10, verbose=2, num_workers=0, callbacks=None):

评估模型。

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
    import numpy as np
    from paddle.incubate.hapi.metrics import Accuracy
    from paddle.incubate.hapi.datasets import MNIST
    from paddle.incubate.hapi.vision.transforms import Compose,Resize
    from paddle.incubate.hapi.vision.models import LeNet
    from paddle.incubate.hapi.model import Input, set_device


    inputs = [Input([-1, 1, 28, 28], 'float32', name='image')]
    labels = [Input([None, 1], 'int64', name='label')]

    val_dataset = MNIST(mode='test')

    model = LeNet()
    model.prepare(metrics=Accuracy(), inputs=inputs, labels=labels)

    result = model.evaluate(val_dataset, batch_size=64)
    print(result)

    # imperative mode
    import paddle.fluid.dygraph as dg
    place = set_device('cpu')
    with dg.guard(place) as g:
        model = LeNet()
        model.prepare(metrics=Accuracy(), inputs=inputs, labels=labels)

        result = model.evaluate(val_dataset, batch_size=64)
        print(result)


.. py:function:: predict(test_data, batch_size=1, num_workers=0, stack_outputs=False, callbacks=None):

模型预测。

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
    from paddle.incubate.hapi.metrics import Accuracy
    from paddle.incubate.hapi.datasets import MNIST
    from paddle.incubate.hapi.vision.transforms import Compose,Resize
    from paddle.incubate.hapi.vision.models import LeNet
    from paddle.incubate.hapi.model import Input, set_device

    class MnistDataset(MNIST):
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

    inputs = [Input([-1, 1, 28, 28], 'float32', name='image')]

    test_dataset = MnistDataset(mode='test', return_label=False)

    model = LeNet()
    model.prepare(inputs=inputs)

    result = model.predict(test_dataset, batch_size=64)
    print(result)

    # imperative mode
    import paddle.fluid.dygraph as dg
    place = set_device('cpu')
    with dg.guard(place) as g:
        model = LeNet()
        model.prepare(inputs=inputs)

        result = model.predict(test_dataset, batch_size=64)
        print(result)


.. py:function:: save_inference_model(save_dir, model_filename=None, params_filename=None, model_only=False):

模型预测。

参数：
    - **save_dir** (str) - 保存推理模型的路径。
    - **model_filename** (str，可选) - 保存预测模型结构 ``Inference Program`` 的文件名称。若设置为None，则使用 ``__model__`` 作为默认的文件名。默认值：None。
    - **params_filename** (str，可选) - 保存预测模型所有相关参数的文件名称。若设置为None，则模型参数被保存在单独的文件中。
    - **model_only** (bool，可选) - 若为True，则只保存预测模型的网络结构，而不保存预测模型的网络参数。默认值：False。

返回：None

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    from paddle.incubate.hapi.model import Model, Input

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self._fc = fluid.dygraph.Linear(784, 1, act='softmax')
        def forward(self, x):
            y = self._fc(x)
            return y

    model = MyModel()
    inputs = [Input([-1, 1, 784], 'float32', name='input')]
    model.prepare(inputs=inputs)

    model.save_inference_model('checkpoint/test')