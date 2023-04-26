.. _cn_api_amp_auto_cast:

auto_cast
-------------------------------

.. py:function:: paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O1', dtype='float16')


创建一个上下文环境，来支持动态图模式下执行的算子的自动混合精度策略（AMP）。
如果启用 AMP，使用 autocast 算法确定每个算子的输入数据类型（float32、float16 或 bfloat16），以获得更好的性能。
通常，它与 ``decorate`` 和 ``GradScaler`` 一起使用，来实现动态图模式下的自动混合精度。


参数
:::::::::
    - **enable** (bool，可选) - 是否开启自动混合精度。默认值为 True。
    - **custom_white_list** (set|list，可选) - 飞桨有默认的白名单，通常不需要设置自定义白名单。自定义白名单中的算子在计算时会被认为是数值安全的，并且对性能至关重要。如果设置了该名单，其中的算子会使用 float16/bfloat16 计算。
    - **custom_black_list** (set|list，可选) - 飞桨有默认的黑名单，可以根据模型特点设置自定义黑名单。自定义黑名单中的算子在计算时会被认为是数值危险的，它们的影响也可能会在下游算子中观察到。该名单中的算子不会转为 float16/bfloat16 计算。
    - **level** (str，可选) - 混合精度训练的优化级别，可为 ``O1`` 、``O2`` 或者 ``OD`` 模式。在 O1 级别下，在白名单中的算子将使用 float16/bfloat16 计算，在黑名单中的算子将使用 float32 计算。在 O2 级别下，模型的参数需要调用 ``decorate`` 转换为float16/bfloat16， 如果算子的浮点型输入全是 float16/bfloat16，算子才会采用 float16/bfloat16 计算，若任意浮点型输入是 float32 类型，算子将采用 float32 计算。在 OD 级别下，飞桨默认的白名单中的算子，如卷积和矩阵乘运算使用 float16/bfloat16 计算，其他算子均使用 float32计算。默认为 O1。
    - **dtype** (str，可选) - 使用的数据类型，可以是 float16 或 bfloat16。默认为 float16。


代码示例
:::::::::
COPY-FROM: paddle.amp.auto_cast
