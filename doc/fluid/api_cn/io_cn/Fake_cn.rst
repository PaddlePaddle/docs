.. _cn_api_fluid_io_Fake:

Fake
-------------------------------

.. py:class:: paddle.fluid.io.Fake

Fakereader将缓存它读取的第一个数据，并将其输出data_num次。它用于缓存来自真实reader的数据，并将其用于速度测试。

参数：
    - **reader** – 原始读取器。
    - **data_num** – reader产生数据的次数 。

返回： 一个Fake读取器


**代码示例**

..  code-block:: python

    def reader():
        for i in range(10):
            yield i

    fake_reader = Fake()(reader, 100)