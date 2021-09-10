.. _cn_api_empty_cache:

empty_cache
-------------------------------

.. py:function:: paddle.device.cuda.empty_cache()


该函数用于释放显存分配器中空闲的显存，这样其他的 GPU 应用程序就可以使用释放出来的显存，并在 ``nvidia-smi`` 中可见。
大多数情况下您不需要使用该函数，当您删除 GPU 上的 Tensor 时，Paddle 并不会将内存释放，而是将显存保留起来，以便在下一次申明显存时可以更快的完成分配。

代码示例
:::::::::
.. code-block:: python

    import paddle

    # required: gpu
    paddle.set_device("gpu")
    # nvidia-smi
    tensor = paddle.randn([512, 512, 512], "float")
    del tensor
    # nvidia-smi
    paddle.device.cuda.empty_cache()
