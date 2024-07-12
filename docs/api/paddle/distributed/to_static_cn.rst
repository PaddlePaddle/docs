.. _cn_api_paddle_distributed_to_static:

to_static
-------------------------------

.. py:function:: paddle.distributed.to_static(layer, loader, loss=None, optimizer=None, strategy=None)

将带有分布式切分信息的动态图 ``layer`` 转换为静态图分布式模型, 可在静态图模式下进行分布式训练；同时将动态图下所使用的数据迭代器 ``loader`` 转换为静态图分布式训练所使用的数据迭代器。

``paddle.distributed.to_static`` 返回 ``DistModel`` 实例和 ``DistributedDataLoader`` 实例。 ``DistModel`` 实例包含了转换后的静态图模型，同时提供了训练、评估和预测的接口。 ``DistributedDataLoader`` 实例用于在静态图分布式训练中加载数据。


参数
:::::::::

    - **layer** (paddle.nn.Layer) - 带有分布式信息，可在动态图模式下进行分布式训练的模型。
    - **loader** (paddle.io.DataLoader) - 动态图训练时所使用的数据迭代器。
    - **loss** (Loss|Callable|None, 可选) - 损失函数。需要训练或者评估模型时，该参数必须设定。
    - **optimizer** (Optimizer|None, 可选) - 优化器。训练模型时，该参数必须设定。
    - **strategy** (Strategy|None, 可选) - 分布式训练的配置，用于设置混合精度训练、分布式优化策略等。

返回
:::::::::
DistModel: 用于静态图分布式训练的模型，通过 ``__call__`` 方法进行训练、评估和预测。需要执行训练、评估或预测时，需要先使用 ``DistModel`` 实例的 ``train()/eval()/predict()`` 方法将其转换为对应的模式。  ``DistModel`` 实例的默认模式会根据 ``paddle.distributed.to_static`` 的输入设置，当 ``loss`` 和 ``optimizer`` 均给定时，默认模式为 ``train``；当 ``optimizer`` 为空时，默认模式为 ``eval``；当 ``loss`` 和 ``optimizer`` 均为空时，默认模式为 ``predict``。

DistributedDataLoader: 用于静态图分布式训练的数据迭代器，和 ``paddle.io.DataLoader`` 用法一致。


代码示例
:::::::::

COPY-FROM: paddle.distributed.to_static
