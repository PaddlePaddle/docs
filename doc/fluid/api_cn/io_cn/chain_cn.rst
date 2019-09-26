.. _cn_api_fluid_io_chain:

chain
-------------------------------

.. py:function:: paddle.fluid.io.chain(*readers)

创建一个数据读取器，其功能是将输入的多个数据读取器的输出链接在一起作为它的输出。

举例来说，如果输入数据读取器的输出分别为[0，0，0]、[10，10，10]和[20，20，20]，那么调用该接口产生的新数据读取器的输出为：[0，0，0][10，10，10][20，20，20] 。

参数：
    - **readers** – 输入的数据读取器。

返回： 新的数据读取器。

返回类型：callable

**代码示例**

..  code-block:: python

    import paddle
    def reader_creator_3(start):
        def reader():
            for i in range(start, start + 3):
                yield [i, i, i]
        return reader

    c = paddle.reader.chain(reader_creator_3(0), reader_creator_3(10), reader_creator_3(20))
    for e in c():
        print(e)
    # 输出结果如下：
    # [0, 0, 0]
    # [1, 1, 1]
    # [2, 2, 2]
    # [10, 10, 10]
    # [11, 11, 11]
    # [12, 12, 12]
    # [20, 20, 20]
    # [21, 21, 21]
    # [22, 22, 22]

