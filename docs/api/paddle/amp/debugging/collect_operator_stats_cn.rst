.. _cn_api_amp_debugging_collect_operator_stats:

collect_operator_stats
-------------------------------

.. py:function:: paddle.amp.debugging.collect_operator_stats()


上下文切换器能够收集不同数据类型的运算符数量。按照float32、float16、bfloat16等四种数据类型进行分类统计数据算子调用次数，在退出context时打印。

**代码示例**：
.. code-block:: python

    import paddle
    
    conv = paddle.nn.Conv2D(3, 2, 3)
    x = paddle.rand([10, 3, 32, 32])
    
    with paddle.amp.debugging.collect_operator_stats():
        # AMP list including conv2d, elementwise_add, reshape2, cast (transfer_dtype)
        with paddle.amp.auto_cast(enable=True, level='O2'):
            out = conv(x)
    # Print to the standard output.
    # <------------------------------------------------------- op list -------------------------------------------------------->
    # <--------------- Op Name ---------------- | -- FP16 Calls --- | -- BF16 Calls --- | --- FP32 Calls--- | -- Other Calls -->
    #   conv2d                                  |  1                |  0                |  0                |  0
    #   elementwise_add                         |  1                |  0                |  0                |  0
    #   reshape2                                |  1                |  0                |  0                |  0
    #   transfer_dtype                          |  0                |  0                |  3                |  0
    # <----------------------------------------------------- op count: 4 ------------------------------------------------------>
    
