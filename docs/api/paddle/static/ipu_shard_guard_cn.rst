.. _cn_api_fluid_ipu_shard_guard:

ipu_shard_guard
-------------------------------

.. py:function:: paddle.static.ipu_shard_guard(index=None, stage=None)


该接口用于对模型进行切分。用于指定Op在哪个ipu上进行计算以及模型被切分之后的计算顺序。

.. note:

仅支持当enable_manual_shard=True, index才能被置为非None。请参阅 :ref:`cn_api_fluid_IpuStrategy` 。
仅支持当enable_pipelining=True, stage才能被置为非None。请参阅 :ref:`cn_api_fluid_IpuStrategy` 。
一个index支持对应None stage或一个stage，一个stage仅支持对应一个新的index或者一个重复的index。

参数
:::::::::
    - **index** (int, 可选) - 指定Op在哪个ipu上计算，（如‘0, 1, 2, 3’），默认值None，表示Op默认跑在IPU 0。
    - **stage** (int, 可选) – 指定被切分的模型的计算顺序，（如‘0, 1, 2, 3’），按照数值大小顺序对被切分的模型进行计算，默认值None，表示没有数据流水计算顺序并按照计算图顺序计算Op。

返回
:::::::::
    无

代码示例
::::::::::

COPY-FROM: paddle.static.ipu_shard_guard