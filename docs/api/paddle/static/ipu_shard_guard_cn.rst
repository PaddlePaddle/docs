.. _cn_api_paddle_static_ipu_shard_guard:

ipu_shard_guard
-------------------------------

.. py:function:: paddle.static.ipu_shard_guard(index=-1, stage=-1)


对模型进行切分。用于指定 Op 在哪个 ipu 上进行计算以及模型被切分之后的计算顺序。

.. note::
    仅支持当 enable_manual_shard=True，才能将 index 设置为非-1 的值。请参阅 :ref:`cn_api_paddle_static_IpuStrategy` 。
    仅支持当 enable_pipelining=True，才能将 stage 设置为非-1 的值。请参阅 :ref:`cn_api_paddle_static_IpuStrategy` 。
    一个 index 支持对应 None stage 或一个 stage，一个 stage 仅支持对应一个新的 index 或者一个重复的 index。

参数
:::::::::
    - **index** (int，可选) - 指定 Op 在哪个 ipu 上计算，（如‘0, 1, 2, 3’），默认值-1，表示 Op 仅在 ipu 0 上运行。
    - **stage** (int，可选) - 指定被切分的模型的计算顺序，（如‘0, 1, 2, 3’），按照数值大小顺序对被切分的模型进行计算，默认值-1，表示没有数据流水计算顺序并按照计算图顺序计算 Op。

返回
:::::::::
    无。

代码示例
::::::::::

COPY-FROM: paddle.static.ipu_shard_guard
