.. _cn_api_paddle_distributed_DistModel:

DistModel
-------------------------------

.. py:class:: paddle.distributed.DistModel()

.. note::
    不推荐直接使用这个 API 生成实例，建议使用 ``paddle.distributed.to_static``。

DistModel 是一个由 ``paddle.nn.Layer`` 转换而来的静态图模型，其中的参数是分布式张量。DistModel 包含了一个由 ``paddle.nn.Layer`` 转换而来的静态图，并提供了模型训练、评估和推理的 API。

DistModel 通过自身的 ``__call__`` 方法来执行模型训练、评估和推理。在执行对应的流程前，请先使用 ``train()/eval()/predict()`` 方法将 DistModel 设置为对应的模式。

详细的使用方法请参考 :ref:`cn_api_paddle_distributed_to_static` API 中的示例代码。

参数
:::::::::

    - **layer** (paddle.nn.Layer) - 动态图中所使用的 ``paddle.nn.Layer`` 实例，其参数包含由 ``paddle.distributed.shard_tensor`` 生成的分布式张量。

    - **loader** (paddle.io.DataLoader) - 动态图模式下所使用的 ``paddle.io.DataLoader`` 实例，用于生成静态图训练所需要的 ``DistributedDataloader``。

    - **loss** (Loss|Callable|None, 可选) - 损失函数。可以是 ``paddle.nn.Layer`` 实例或任何可调用函数。如果 loss 不为 None，则 DistModel 会默认设置为 "train"（当 optimizer 不为 None 时）或 "eval" 模式（当 optimizer 为 None 时）。如果 loss 为 None，则 DistModel 会默认设置为 "predict" 模式。默认值：None。

    - **optimizer** (paddle.optimizer.Optimizer|None, 可选) - 优化器。如果同时设置了 optimizer 和 loss，DistModel 会默认设置为 "train" 模式。默认值：None。

    - **strategy** (paddle.distributed.Strategy|None, 可选) - 并行策略和优化策略的配置（例如优化器分片、流水线并行等）。默认值：None。


**代码示例**

COPY-FROM: paddle.distributed.to_static


方法
:::::::::


train()
''''''''

将 DistModel 设置为 "train" 模式。在 "train" 模式下，执行 ``__call__`` 方法会更新模型的参数并返回 loss。


eval()
'''''''

将 DistModel 设置为 "eval" 模式。在 "eval" 模式下，执行 ``__call__`` 方法会返回 loss。


predict()
'''''''

将 DistModel 设置为 "predict" 模式。在 "predict" 模式下，执行 ``__call__`` 方法会以字典的格式返回模型的输出。


dist_main_program(mode=None)
'''''''

获取指定 ``mode`` 的分布式 ``main_program``。 ``main_program`` 用于存储模型中变量和计算的描述信息。分布式 ``main_program`` 存储了每个进程上局部模型的描述信息。每个 ``mode`` 都有自己的分布式 ``main_program``，用于执行模型的训练、评估或推理。 ``dist_main_program`` 返回指定 ``mode`` 的对应分布式 ``main_program``。

**参数**

    - **mode** (str|None, 可选) - 指定需要返回的 ``main_program`` 的模式，可以是 "train"、"eval" 或 "predict"，如果未设置，则使用 DistModel 的当前模式。默认值：None。

**返回**

    指定 ``mode`` 的分布式 ``main_program``。


dist_startup_program(mode=None)
'''''''

获取指定 ``mode`` 的分布式 ``startup_program``。 ``startup_program`` 用于存储模型初始化的描述信息。分布式 ``startup_program`` 存储了每个进程上局部模型的初始化操作。每个 ``mode`` 都有自己的分布式 ``startup_program``，用于模型训练、评估或推理时的初始化。

**参数**

    - **mode** (str|None, 可选) - 指定需要返回的 ``startup_program`` 的模式，可以是 "train"、 "eval" 或 "predict"，如果未设置，则使用 DistModel 的当前模式。默认值：None。

**返回**

    指定 ``mode`` 的分布式 ``startup_program``。


serial_main_program(mode=None)
'''''''

获取指定 ``mode`` 的串行 ``main_program``。串行 ``main_program`` 是包含了完整模型的变量和计算描述信息。

**参数**

    - **mode** (str|None, 可选) - 指定需要返回的 ``main_program`` 的模式，可以是 "train"、 "eval" 或 "predict"，如果未设置，则使用 DistModel 的当前模式。默认值：None。

**返回**

    指定 ``mode`` 的串行 ``main_program``。


serial_startup_program(mode=None)
'''''''

获取指定 ``mode`` 的串行 ``startup_program``，包含了完整的初始化操作。

**参数**

    - **mode** (str|None, 可选) - 指定需要返回的 ``startup_program`` 的模式，可以是 "train "、"eval" 或 "predict"，如果未设置，则使用 DistModel 的当前模式。默认值：None。

**返回**

    指定 ``mode`` 的串行 ``startup_program``。
