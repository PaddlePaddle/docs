.. _cn_api_fluid_io_firstn:

firstn
-------------------------------

.. py:function:: paddle.fluid.io.firstn(reader, n)

限制reader可以返回的最大样本数。

参数：
    - **reader** (callable)  – 要读取的数据读取器。
    - **n** (int)  – 返回的最大样本数 。

返回： 装饰reader

返回类型： callable