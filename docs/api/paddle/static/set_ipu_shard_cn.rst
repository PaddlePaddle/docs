.. _cn_api_paddle_static_set_ipu_shard:

set_ipu_shard
-------------------------------

.. py:function:: paddle.static.set_ipu_shard(call_func, index=-1, stage=-1)


通过设置输入的函数或计算层内每个算子的流水线属性实现对模型的切分。

.. note::
    仅支持当 enable_manual_shard=True，才能将 index 设置为非-1 的值。请参阅 :ref:`cn_api_paddle_static_IpuStrategy` 。
    仅支持当 enable_pipelining=True，才能将 stage 设置为非-1 的值。请参阅 :ref:`cn_api_paddle_static_IpuStrategy` 。
    一个 index 支持对应 None stage 或一个 stage，一个 stage 仅支持对应一个新的 index 或者一个重复的 index。

参数
:::::::::
    - **call_func** (Layer|function) - 静态图下的函数或者计算层。
    - **index** (int，可选) - 指定 Op 在哪个 ipu 上计算，（如‘0, 1, 2, 3’），默认值-1，表示 Op 仅在 ipu 0 上运行。
    - **stage** (int，可选) - 指定被切分的模型的计算顺序，（如‘0, 1, 2, 3’），按照数值大小顺序对被切分的模型进行计算，默认值-1，表示没有数据流水计算顺序并按照计算图顺序计算 Op。

返回
:::::::::
    包装后的调用函数。

代码示例
::::::::::

COPY-FROM: paddle.static.set_ipu_shard
