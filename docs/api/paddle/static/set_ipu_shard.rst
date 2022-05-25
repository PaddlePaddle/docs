.. _cn_api_fluid_set_ipu_shard:

set_ipu_shard
-------------------------------

.. py:function:: paddle.static.set_ipu_shard(call_func, index=-1, stage=-1)


该接口通过设置输入的函数或计算层内每个算子的流水线属性实现对模型的切分。

.. note:

仅支持当enable_manual_shard=True, index设置才有效。请参阅 :ref:`cn_api_fluid_IpuStrategy` 。
仅支持当enable_pipelining=True, stage设置才有效。请参阅 :ref:`cn_api_fluid_IpuStrategy` 。
一个index支持对应None stage或一个stage，一个stage仅支持对应一个新的index或者一个重复的index。

参数
:::::::::
    - **call_func** (Layer|function) - 静态图下的函数或者计算层。
    - **index** (int, 可选) - 指定Op在哪个ipu上计算，（如‘0, 1, 2, 3’），默认值-1，表示不指定ipu。
    - **stage** (int, 可选) – 指定被切分的模型的计算顺序，（如‘0, 1, 2, 3’），按照数值大小顺序对被切分的模型进行计算，默认值-1，表示没有数据流水计算顺序并按照计算图顺序计算Op。

返回
:::::::::
    无。

代码示例
::::::::::

COPY-FROM: paddle.static.set_ipu_shard