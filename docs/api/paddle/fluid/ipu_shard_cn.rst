.. _cn_api_fluid_ipu_shard:

ipu_shard
-------------------------------

.. py:function:: paddle.fluid.ipu_shard(ipu_index=None, ipu_stage=None)


该接口用于对模型进行切分。用于指定tensor在哪个ipu上进行计算以及模型被切分之后的计算顺序。该接口无返回值。

参数
:::::::::
    - **ipu_index** (int, optional) - 指定Tensor在哪个ipu上计算，（如‘0、1’），默认值None。
    - **ipu_stage** (int, optional) – 指定被切分的模型的计算顺序，（如‘0、1’），默认值None。按照数值大小对被切分的模型进行计算。

返回
:::::::::
    无

代码示例
::::::::::

COPY-FROM: paddle.fluid.ipu_shard
