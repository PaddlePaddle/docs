.. _cn_api_fluid_io_deserialize_persistables:

deserialize_persistables
-------------------------------


.. py:function:: paddle.static.deserialize_persistables(program, data, executor)




根据指定的 program 和 executor，反序列化模型参数。

参数
::::::::::::

  - **program** (Program) - 指定包含要反序列化的参数的名称的 program。
  - **data** (bytes) - 序列化之后的模型参数。
  - **executor** (Executor) - 用来执行 load op 的 ``executor`` 。

返回
::::::::::::

  - Program：包含反序列化后的参数的 program。

代码示例
::::::::::::

COPY-FROM: paddle.static.deserialize_persistables
