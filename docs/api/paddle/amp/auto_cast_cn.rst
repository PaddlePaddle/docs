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
COPY-FROM: paddle.amp.auto_cast

