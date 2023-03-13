.. _cn_api_fluid_layers_read_file:

read_file
-------------------------------


.. py:function:: paddle.fluid.layers.read_file(reader)




从给定的 reader 中读取数据

reader 是一个 Variable，它可以是由函数 fluid.layers.py_reader()生成的 reader，或者是由函数 fluid.layers.double_buffer()生成的装饰 Variable。

参数
::::::::::::

    - **reader** (Variable)-待处理的 reader

返回
::::::::::::
从 reader 中读取的数据元组，元组数据类型为 Variable

返回类型
::::::::::::
 tuple（元组）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.read_file
