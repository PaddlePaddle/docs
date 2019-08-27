.. _cn_api_fluid_io_chain:

chain
-------------------------------

.. py:function:: paddle.fluid.io.chain(*readers)

创建一个数据读取器，输出为输入数据读取器链接到一起的结果，如果输入如下：

[0, 0, 0]

[1, 1, 1]

[2, 2, 2]

输出将会为[0, 0, 0, 1, 1, 1, 2, 2, 2].

参数:
    - **readers** – 输入reader

返回：新的数据reader。

返回类型：callable