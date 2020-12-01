.. _cn_api_amp_auto_cast:

auto_cast
-------------------------------

.. py:function:: paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None)


创建一个上下文环境，来支持动态图模式下执行的算子的自动混合精度策略（AMP）。
如果启用AMP，使用autocast算法确定每个算子的输入数据类型（float32或float16），以获得更好的性能。
通常，它与 ``GradScaler`` 一起使用，来实现动态图模式下的自动混合精度。


参数：
    - **enable** (bool, 可选) - 是否开启自动混合精度。默认值为True。
    - **custom_white_list** (set|list, 可选) - 自定义算子白名单。这个名单中的算子在支持float16计算时会被认为是数值安全的，并且对性能至关重要。如果设置了白名单，该名单中的算子会使用float16计算。
    - **custom_black_list** (set|list, 可选) - 自定义算子黑名单。这个名单中的算子在支持float16计算时会被认为是数值危险的，它们的影响也可能会在下游操作中观察到。这些算子通常不会转为float16计算。


**代码示例**：

.. code-block:: python

    import paddle

    conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
    data = paddle.rand([10, 3, 32, 32])

    with paddle.amp.auto_cast():
        conv = conv2d(data)
        print(conv.dtype) # FP16

    with paddle.amp.auto_cast(enable=False):
        conv = conv2d(data)
        print(conv.dtype) # FP32

    with paddle.amp.auto_cast(custom_black_list={'conv2d'}):
        conv = conv2d(data)
        print(conv.dtype) # FP32

    a = paddle.rand([2,3])
    b = paddle.rand([2,3])
    with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}):
        c = a + b
        print(c.dtype) # FP16




