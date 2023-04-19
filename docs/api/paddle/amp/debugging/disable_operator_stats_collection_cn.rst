.. _cn_api_amp_debugging_disable_operator_stats_collection():

paddle.amp.debugging.disable_operator_stats_collection()
-------------------------------

.. py:function:: paddle.amp.debugging.disable_operator_stats_collection()


禁用收集不同数据类型的算子调用次数。该函数与相应的使能函数配对使用。按照float32、float16、bfloat16等四种数据类型进行分类统计算子调用次数。

**代码示例**：
.. code-block:: python

    import paddle
    
    conv = paddle.nn.Conv2D(3, 2, 3)
    x = paddle.rand([10, 3, 32, 32])
    
    paddle.amp.debugging.enable_operator_stats_collection()
    # AMP list including conv2d, elementwise_add, reshape2, cast (transfer_dtype)
    with paddle.amp.auto_cast(enable=True, level='O2'):
        out = conv(x)
    # Print to the standard output.
    paddle.amp.debugging.disable_operator_stats_collection()
    # <------------------------------------------------------- op list -------------------------------------------------------->
    # <--------------- Op Name ---------------- | -- FP16 Calls --- | -- BF16 Calls --- | --- FP32 Calls--- | -- Other Calls -->
    #   conv2d                                  |  1                |  0                |  0                |  0
    #   elementwise_add                         |  1                |  0                |  0                |  0
    #   reshape2                                |  1                |  0                |  0                |  0
    #   transfer_dtype                          |  0                |  0                |  3                |  0
    # <----------------------------------------------------- op count: 4 ------------------------------------------------------>

