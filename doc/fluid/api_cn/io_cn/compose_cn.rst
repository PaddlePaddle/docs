.. _cn_api_fluid_io_compose:

compose
-------------------------------

.. py:function:: paddle.fluid.io.compose(*readers, **kwargs)

该接口将多个数据读取器组合为一个数据读取器，返回读取器的输出包含所有输入读取器的输出。

例如：如果输入为三个reader，三个reader的输出分别为：（1，2）、3、（4，5），则组合reader的输出为：（1，2，3，4，5）。

参数：
    - **readers** - 将被组合的多个读取器。
    - **check_alignment** (bool) - 可选，指明是否对输入reader进行对齐检查，默认值为True。如果为True，将检查输入reader是否正确对齐。如果为False，将不检查对齐并自动丢弃无法对齐的末尾数据。

返回：数据读取器。

**代码示例**:

.. code-block:: python

     import paddle.io
     reader1 = data_reader()
     reader2 = data_reader()
     reader_compose = paddle.io.compose(reader1, reader2, check_alignment=False)

注意： 运行过程可能会抛出异常 ``ComposeNotAligned`` ，原因是输入的readers数据未对齐。 当check_alignment设置为False时，不会检查并触发该异常。
