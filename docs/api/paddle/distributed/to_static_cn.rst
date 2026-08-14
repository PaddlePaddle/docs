.. _cn_api_paddle_distributed_to_static:

to_static
-------------------------------

.. py:function:: paddle.distributed.to_static(layer, loader=None, loss=None, optimizer=None, strategy=None, input_spec=None)

将带有分布式切分信息的动态图 ``layer`` 转换为静态图分布式模型，可在静态图模式下进行分布式训练、评估和预测。

``paddle.distributed.to_static`` 返回包含转换后静态图模型的 ``DistModel`` 实例，并提供训练、评估和预测接口。


参数
:::::::::

    - **layer** (paddle.nn.Layer) - 带有分布式信息，可在动态图模式下进行分布式训练的模型。
    - **loader** (ShardDataloader|paddle.io.DataLoader|None，可选) - 动态图模式下使用的数据加载器，用于推断 input_spec 和 label_spec。默认值为 None。
    - **loss** (Loss|Callable|None，可选) - 损失函数。需要训练或者评估模型时，该参数必须设定。
    - **optimizer** (Optimizer|_ShardOptimizer|None，可选) - 优化器。可以是 ``paddle.optimizer.Optimizer``，也可以是由 ``shard_optimizer`` 包装的 ``_ShardOptimizer``。训练模型时，该参数必须设定。
    - **strategy** (Strategy|None，可选) - 分布式训练的配置，用于设置混合精度训练、分布式优化策略等。
    - **input_spec** (list[list[paddle.distributed.DistributedInputSpec]]|None，可选) - 自定义输入规格，指定模型输入和标签的形状、数据类型及名称信息。非 None 时，由该参数推断输入和标签规格；其应包含两个子列表，第一个表示输入规格，第二个表示标签规格。默认值为 None。

返回
:::::::::
DistModel: 用于静态图分布式训练的模型，通过 ``__call__`` 方法进行训练、评估和预测。需要执行训练、评估或预测时，需要先使用 ``DistModel`` 实例的 ``train()/eval()/predict()`` 方法将其转换为对应的模式。``DistModel`` 实例的默认模式会根据 ``paddle.distributed.to_static`` 的输入设置：当 ``loss`` 和 ``optimizer`` 均给定时，默认模式为 ``train``；当 ``loss`` 非空且 ``optimizer`` 为空时，默认模式为 ``eval``；当 ``loss`` 为空时，默认模式为 ``predict``。


代码示例
:::::::::

COPY-FROM: paddle.distributed.to_static
