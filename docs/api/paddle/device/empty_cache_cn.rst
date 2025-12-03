.. _cn_api_paddle_device_empty_cache:

empty_cache
-----------

.. py:function:: paddle.device.empty_cache()

释放当前设备上所有未占用的缓存内存。

代码示例
::::::::::::
.. code-block:: python

    import paddle

    x = paddle.randn([1000, 1000])
    del x
    paddle.device.empty_cache()
