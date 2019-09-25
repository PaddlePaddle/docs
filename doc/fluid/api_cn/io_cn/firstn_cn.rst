.. _cn_api_fluid_io_firstn:

firstn
-------------------------------

.. py:function:: paddle.fluid.io.firstn(reader, n)

该接口创建一个数据读取器，它可以返回的最大样本数为n。

参数：
    - **reader** (callable)  – 输入的数据读取器。
    - **n** (int)  – 可以返回的最大样本数。

返回： 新的的数据读取器。

返回类型： callable

..  code-block:: python

    import paddle.fluid as fluid
    def reader():
        for i in range(100):
            yield i
    firstn_reader = fluid.io.firstn(reader, 5)
    for e in firstn_reader():
        print(e)
    # 输出结果为:0 1 2 3 4
