.. _cn_api_fluid_io_batch:

batch
-------------------------------

.. py:function:: paddle.fluid.io.batch(reader, batch_size, drop_last=False)

该API是一个reader的装饰器。返回的reader将输入reader的数据打包成指定的batch_size大小的批处理数据（batched data）。

参数：
    - **reader** (Variable)- 读取数据的数据reader。
    - **batch_size** (int)- 批尺寸。
    - **drop_last** (bool) - 如果最后一个batch不等于batch_size，则放弃最后一个batch。默认值为False。

返回：batched reader

返回类型：callable

**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    import paddle.dataset.mnist as mnist
    
    batch_reader = fluid.io.batch(mnist.train(), batch_size=5)

