.. _cn_api_paddle_Model:

Model
-------------------------------

.. py:class:: paddle.Model(network, inputs=None, labels=None)

``Model`` 对象是一个具备训练、测试、推理的神经网络。该对象同时支持静态图和动态图模式，飞桨框架默认为动态图模式，通过 ``paddle.enable_static()`` 来切换到静态图模式。需要注意的是，需要在实例化 ``Model`` 对象之前完成切换。

在 GPU 上训练时，高层 API 支持自动混合精度（AMP）训练，并且在静态图下使用 Adam、AdamW、Momentum 优化器时还支持纯 float16 的训练。在使用纯 float16 训练之前，优化器初始化时 ``multi_precision`` 参数可以设置为 True，这样可以避免性能变差或是收敛变慢的问题。并且，在组网中可以使用 ``paddle.static.amp.fp16_guard`` 来限定使用纯 float16 训练的范围，否则需要把 ``use_fp16_guard`` 手动设置为 False 以开启全局纯 float16 训练。使用纯 float16 训练前，可能需要手动将 dtype 为 float32 的输入转成 float16 的输入。然而，使用自动混合精度训练（AMP）时，不支持限定混合精度训练的范围。

参数
:::::::::

    - **network** (paddle.nn.Layer) - 是 ``paddle.nn.Layer`` 的一个实例。
    - **inputs** (InputSpec|list|tuple|dict|None，可选) - ``network`` 的输入，可以是 ``InputSpec`` 的实例，或者是一个 ``InputSpec`` 的 ``list``，或者是格式为 ``{name: InputSpec}`` 的 ``dict``，或者为 ``None``。默认值为 ``None``。
    - **labels** (InputSpec|list|tuple|None，可选) - ``network`` 的标签，可以是 ``InputSpec`` 的实例，或者是一个 ``InputSpec`` 的 ``list``，或者为 ``None``。 默认值为 ``None``。

.. note::

    在动态图中，``inputs`` 和 ``labels`` 都可以设置为 ``None``。但是，在静态图中，``input`` 不能设置为 ``None``。而如果损失函数需要标签（label）作为输入，则必须设置 ``labels``，否则，可以为 ``None``。


代码示例
:::::::::

1. 一般示例

COPY-FROM: paddle.Model:code-example1


2. 使用混合精度训练的例子

COPY-FROM: paddle.Model:code-example2


方法
:::::::::

train_batch(inputs, labels=None, update=True)
'''''''''

在一个批次的数据上进行训练。

**参数**

    - **inputs** (numpy.ndarray|Tensor|list) - 一批次的输入数据。它可以是一个 numpy 数组或 paddle.Tensor，或者是它们的列表（在模型具有多输入的情况下）。
    - **labels** (numpy.ndarray|Tensor|list，可选) - 一批次的标签。它可以是一个 numpy 数组或 paddle.Tensor，或者是它们的列表（在模型具有多输入的情况下）。如果无标签，请设置为 None。默认值：None。
    - **update** (bool，可选) - 是否在 loss.backward() 计算完成后更新参数，将它设置为 False 可以累积梯度。默认值：True。


**返回**

如果没有定义评估函数，则返回包含了训练损失函数的值的列表；如果定义了评估函数，则返回一个元组（损失函数的列表，评估指标的列表）。


**代码示例**

COPY-FROM: paddle.Model.train_batch


eval_batch(inputs, labels=None)
'''''''''

在一个批次的数据上进行评估。

**参数**


    - **inputs** (numpy.ndarray|Tensor|list) - 一批次的输入数据。它可以是一个 numpy 数组或 paddle.Tensor，或者是它们的列表（在模型具有多输入的情况下）。
    - **labels** (numpy.ndarray|Tensor|list，可选) - 一批次的标签。它可以是一个 numpy 数组或 paddle.Tensor，或者是它们的列表（在模型具有多输入的情况下）。如果无标签，请设置为 None。默认值：None。

**返回**

list，如果没有定义评估函数，则返回包含了预测损失函数的值的列表；如果定义了评估函数，则返回一个元组（损失函数的列表，评估指标的列表）。


**代码示例**

COPY-FROM: paddle.Model.eval_batch


predict_batch(inputs)
'''''''''

在一个批次的数据上进行测试。

**参数**


    - **inputs** (numpy.ndarray|Tensor|list) - 一批次的输入数据。它可以是一个 numpy 数组或 paddle.Tensor，或者是它们的列表（在模型具有多输入的情况下）。

**返回**

一个列表，包含了模型的输出。

**代码示例**

COPY-FROM: paddle.Model.predict_batch


save(path, training=True)
'''''''''

将模型的参数和训练过程中优化器的信息保存到指定的路径，以及推理所需的参数与文件。如果 training=True，所有的模型参数都会保存到一个后缀为 ``.pdparams`` 的文件中。
所有的优化器信息和相关参数，比如 ``Adam`` 优化器中的 ``beta1`` ， ``beta2`` ，``momentum`` 等，都会被保存到后缀为 ``.pdopt``。如果优化器比如 SGD 没有参数，则该不会产生该文件。如果 training=False，则不会保存上述说的文件。只会保存推理需要的参数文件和模型文件。

**参数**


    - **path** (str) - 保存的文件名前缀。格式如 ``dirname/file_prefix`` 或者 ``file_prefix`` 。
    - **training** (bool，可选) - 是否保存训练的状态，包括模型参数和优化器参数等。如果为 False，则只保存推理所需的参数与文件。默认值：True。

**返回**

无。

**代码示例**

COPY-FROM: paddle.Model.save


load(path, skip_mismatch=False, reset_optimizer=False)
'''''''''

从指定的文件中载入模型参数和优化器参数，如果不想恢复优化器参数信息，优化器信息文件可以不存在。需要注意的是：参数名称的检索是根据保存模型时结构化的名字，当想要载入参数进行迁移学习时要保证预训练模型和当前的模型的参数有一样结构化的名字。

**参数**


    - **path** (str) - 保存参数或优化器信息的文件前缀。格式如 ``path.pdparams`` 或者 ``path.pdopt`` ，后者是非必要的，如果不想恢复优化器信息。
    - **skip_mismatch** (bool，可选) - 是否需要跳过保存的模型文件中形状或名称不匹配的参数，设置为 ``False`` 时，当遇到不匹配的参数会抛出一个错误。默认值：False。
    - **reset_optimizer** (bool，可选) - 设置为 ``True`` 时，会忽略提供的优化器信息文件。否则会载入提供的优化器信息。默认值：False。

**返回**

无。

**代码示例**

COPY-FROM: paddle.Model.load


parameters(*args, **kwargs)
'''''''''

返回一个包含模型所有参数的列表。

**返回**

在静态图中返回一个包含 ``Parameter`` 的列表，在动态图中返回一个包含 ``ParamBase`` 的列表。

**代码示例**

COPY-FROM: paddle.Model.parameters


prepare(optimizer=None, loss=None, metrics=None, amp_configs=None)
'''''''''

配置模型所需的部件，比如优化器、损失函数和评价指标。

**参数**

    - **optimizer** (OOptimizer|None，可选) - 当训练模型的，该参数必须被设定。当评估或测试的时候，该参数可以不设定。默认值：None。
    - **loss** (Loss|Callable|None，可选) - 当训练模型的，该参数必须被设定。默认值：None。
    - **metrics** (Metric|list[Metric]|None，可选) - 当该参数被设定时，所有给定的评估方法会在训练和测试时被运行，并返回对应的指标。默认值：None。
    - **amp_configs** (str|dict|None，可选) - 混合精度训练的配置，通常是个 dict，也可以是 str。当使用自动混合精度训练或者纯 float16 训练时，``amp_configs`` 的 key ``level`` 需要被设置为 O1 或者 O2，float32 训练时则默认为 O0。除了 ``level`` ，还可以传入更多的和混合精度 API 一致的参数，例如：``init_loss_scaling``、 ``incr_ratio`` 、 ``decr_ratio``、 ``incr_every_n_steps``、 ``decr_every_n_nan_or_inf``、 ``use_dynamic_loss_scaling``、 ``custom_white_list``、 ``custom_black_list`` ，在静态图下还支持传入 ``custom_black_varnames`` 和 ``use_fp16_guard`` 。详细使用方法可以参考参考混合精度 API 的文档 :ref:`auto_cast <cn_api_paddle_amp_auto_cast>`  和 :ref:`GradScaler <cn_api_paddle_amp_GradScaler>` 。为了方便起见，当不设置其他的配置参数时，也可以直接传入 ``'O1'`` 、``'O2'`` 。在使用 float32 训练时，该参数可以为 None。默认值：None。


fit(train_data=None, eval_data=None, batch_size=1, epochs=1, eval_freq=1, log_freq=10, save_dir=None, save_freq=1, verbose=2, drop_last=False, shuffle=True, num_workers=0, callbacks=None, accumulate_grad_batches=1, num_iters=None)
'''''''''

训练模型。当 ``eval_data`` 给定时，会在 ``eval_freq`` 个 ``epoch`` 后进行一次评估。

**参数**

    - **train_data** (Dataset|DataLoader，可选) - 一个可迭代的数据源，推荐给定一个 ``paddle paddle.io.Dataset`` 或 ``paddle.io.Dataloader`` 的实例。默认值：None。
    - **eval_data** (Dataset|DataLoader，可选) - 一个可迭代的数据源，推荐给定一个 ``paddle paddle.io.Dataset`` 或 ``paddle.io.Dataloader`` 的实例。当给定时，会在每个 ``epoch`` 后都会进行评估。默认值：None。
    - **batch_size** (int，可选) - 训练数据或评估数据的批大小，当 ``train_data`` 或 ``eval_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：1。
    - **epochs** (int，可选) - 训练的轮数。默认值：1。
    - **eval_freq** (int，可选) - 评估的频率，多少个 ``epoch`` 评估一次。默认值：1。
    - **log_freq** (int，可选) - 日志打印的频率，多少个 ``step`` 打印一次日志。默认值：10。
    - **save_dir** (str|None，可选) - 保存模型的文件夹，如果不设定，将不保存模型。默认值：None。
    - **save_freq** (int，可选) - 保存模型的频率，多少个 ``epoch`` 保存一次模型。默认值：1。
    - **verbose** (int，可选) - 可视化的模型，必须为 0，1，2。当设定为 0 时，不打印日志，设定为 1 时，使用进度条的方式打印日志，设定为 2 时，一行一行地打印日志。默认值：2。
    - **drop_last** (bool，可选) - 是否丢弃训练数据中最后几个不足设定的批次大小的数据。默认值：False。
    - **shuffle** (bool，可选) - 是否对训练数据进行洗牌。当 ``train_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：True。
    - **num_workers** (int，可选) - 启动子进程用于读取数据的数量。当 ``train_data`` 和 ``eval_data`` 都为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：0。
    - **callbacks** (Callback|list[Callback]|None，可选) -  ``Callback`` 的一个实例或实例列表。该参数不给定时，默认会插入 :ref:`cn_api_paddle_callbacks_ProgBarLogger` 和 :ref:`cn_api_paddle_callbacks_ModelCheckpoint` 这两个实例。默认值：None。
    - **accumulate_grad_batches** (int，可选) -  训练过程中，在优化器更新之前累积梯度的批次数。通过改变该参数可以模拟大的 ``batch_size``。默认值：1。
    - **num_iters** (int，可选) -  训练模型过程中的迭代次数。如果设置为 None，则根据参数 ``epochs`` 来训练模型，否则训练模型 ``num_iters`` 次。默认值：None。


**返回**

无。

**代码示例**

    1. 使用 Dataset 训练，并设置 batch_size 的例子。

    COPY-FROM: paddle.Model.fit:code-example3


    2. 使用 Dataloader 训练的例子.

    COPY-FROM: paddle.Model.fit:code-example4


evaluate(eval_data, batch_size=1, log_freq=10, verbose=2, num_workers=0, callbacks=None, num_iters=None)
'''''''''

在输入数据上，评估模型的损失函数值和评估指标。

**参数**

    - **eval_data** (Dataset|DataLoader) - 一个可迭代的数据源，推荐给定一个 ``paddle.io.Dataset`` 或 ``paddle.io.Dataloader`` 的实例。默认值：None。
    - **batch_size** (int，可选) - 训练数据或评估数据的批大小，当 ``eval_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：1。
    - **log_freq** (int，可选) - 日志打印的频率，多少个 ``step`` 打印一次日志。默认值：10。
    - **verbose** (int，可选) - 可视化的模型，必须为 0，1，2。当设定为 0 时，不打印日志，设定为 1 时，使用进度条的方式打印日志，设定为 2 时，一行一行地打印日志。默认值：2。
    - **num_workers** (int，可选) - 启动子进程用于读取数据的数量。当 ``eval_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：0。
    - **callbacks** (Callback|list[Callback]|None，可选) -  ``Callback`` 的一个实例或实例列表。该参数不给定时，默认会插入 ``ProgBarLogger`` 和 ``ModelCheckpoint`` 这两个实例。默认值：None。
    - **num_iters** (int，可选) -  训练模型过程中的迭代次数。如果设置为 None，则根据参数 ``epochs`` 来训练模型，否则训练模型 ``num_iters`` 次。默认值：None。

**返回**

dict, key 是 ``prepare`` 时 Metric 的的名称，value 是该 Metric 的值。

**代码示例**

COPY-FROM: paddle.Model.evaluate


predict(test_data, batch_size=1, num_workers=0, stack_outputs=False, verbose=1, callbacks=None)
'''''''''

在输入数据上，预测模型的输出。

**参数**

    - **test_data** (Dataset|DataLoader) - 一个可迭代的数据源，推荐给定一个 ``paddle.io.Dataset`` 或 ``paddle.io.Dataloader`` 的实例。默认值：None。
    - **batch_size** (int，可选) - 训练数据或评估数据的批大小，当 ``test_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：1。
    - **num_workers** (int，可选) - 启动子进程用于读取数据的数量。当 ``test_data`` 为 ``DataLoader`` 的实例时，该参数会被忽略。默认值：0。
    - **stack_outputs** (bool，可选) - 是否将输出进行堆叠。比如对于单个样本输出形状为 ``[X, Y]``，``test_data`` 包含 N 个样本的情况，如果 ``stack_outputs`` 设置为 True，那么输出的形状将会是 ``[N, X, Y]``，如果 ``stack_outputs`` 设置为 False，那么输出的形状将会是 ``[[X, Y], [X, Y], ..., [X, Y]]``。将 ``stack_outputs`` 设置为 False 适用于输出为 DenseTensor 的情况，如果输出不包含 DenseTensor，建议将其设置为 True。默认值：False。
    - **verbose** (int，可选) - 可视化的模型，必须为 0，1，2。当设定为 0 时，不打印日志，设定为 1 时，使用进度条的方式打印日志，设定为 2 时，一行一行地打印日志。默认值：1。
    - **callbacks** (Callback|list[Callback]|None，可选) -  ``Callback`` 的一个实例或实例列表。默认值：None。

**返回**

模型的输出。

**代码示例**

COPY-FROM: paddle.Model.predict


summary(input_size=None, dtype=None)
'''''''''

打印网络的基础结构和参数信息。

**参数**

    - **input_size** (tuple|InputSpec|list[tuple|InputSpec]，可选) - 输入 Tensor 的大小。如果网络只有一个输入，那么该值需要设定为 tuple 或 InputSpec。如果模型有多个输入。那么该值需要设定为 list[tuple|InputSpec]，包含每个输入的 shape 。如果该值没有设置，会将 ``self._inputs`` 作为输入。默认值：None。
    - **dtype** (str，可选) - 输入 Tensor 的数据类型，如果没有给定，默认使用 ``float32`` 类型。默认值：None。

**返回**

字典：包含网络全部参数的大小和全部可训练参数的大小。

**代码示例**

COPY-FROM: paddle.Model.summary
