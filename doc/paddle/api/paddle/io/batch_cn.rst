.. _cn_api_fluid_io_batch:

batch
-------------------------------

.. py:function:: paddle.fluid.io.batch(reader, batch_size, drop_last=False)




该接口是一个reader的装饰器。返回的reader将输入reader的数据打包成指定的batch_size大小的批处理数据（batched data）。

参数：
    - **reader** (generator)- 读取数据的数据reader。
    - **batch_size** (int)- 批尺寸。
    - **drop_last** (bool) - 若设置为True，则当最后一个batch不等于batch_size时，丢弃最后一个batch；若设置为False，则不会。默认值为False。

返回：batched reader

返回类型：generator

**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid

    def reader():
        for i in range(10):
            yield i
    batch_reader = fluid.io.batch(reader, batch_size=2)
    
    for data in batch_reader():
        print(data)
    
    # 输出为:
    # [0, 1]
    # [2, 3]
    # [4, 5]
    # [6, 7]
    # [8, 9]        









