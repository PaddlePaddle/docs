.. _cn_api_amp_auto_cast:

auto_cast
-------------------------------

.. py:function:: paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O1')


创建一个上下文环境，来支持动态图模式下执行的算子的自动混合精度策略（AMP）。
如果启用AMP，使用autocast算法确定每个算子的输入数据类型（float32或float16），以获得更好的性能。
通常，它与 ``decorate`` 和 ``GradScaler`` 一起使用，来实现动态图模式下的自动混合精度。
混合精度训练提供两种模式：``O1``代表采用黑名名单策略的混合精度训练；``O2``代表纯float16训练，除自定义黑名单和不支持float16的算子之外，全部使用float16计算。


参数：
    - **enable** (bool, 可选) - 是否开启自动混合精度。默认值为True。
    - **custom_white_list** (set|list, 可选) - 自定义算子白名单。这个名单中的算子在支持float16计算时会被认为是数值安全的，并且对性能至关重要。如果设置了白名单，该名单中的算子会使用float16计算。
    - **custom_black_list** (set|list, 可选) - 自定义算子黑名单。这个名单中的算子在支持float16计算时会被认为是数值危险的，它们的影响也可能会在下游操作中观察到。这些算子通常不会转为float16计算。
    - **level** (str, 可选) - 混合精度训练模式，可为``O1``或``O2``模式，默认``O1``模式。


**代码示例**：
COPY-FROM: paddle.amp.auto_cast


.. py:function:: paddle.amp.decorate(models, optimizers=None, level='O1', master_weight=None, save_dtype=None)


装饰神经网络参数，来支持动态图模式下执行的算子的自动混合精度策略（AMP）。
在``O1``模式下，该函数不做任何处理，直接返回输入的models和optimizers。在``O2``模式下，将对输入的网络参数数据类型由float32转为float16，（除BatchNorm和LayerNorm）。
通过该函数可为支持master weight策略的优化器开启master weight策略，以保证训练精度。通过 ``save_dtype`` 可指定 ``paddle.save`` 和 ``paddle.jit.save`` 存储的网络参数数据类型。


参数：
    - **models** (Layer|list of Layer) - 网络模型。在``O2``模式下，输入的模型参数将由float32转为float16。
    - **optimizers** (Optimizer|list of Optimizer) - 优化器，在``O2``模式下，将同步更新网络更改后的flaot16参数。
    - **level** (str, 可选) - 混合精度训练模式，默认``O1``模式。
    - **master_weight** (bool|None, 可选) - 是否使用master weight策略。支持maser weight策略的优化器包括``adam``、``adamW``、``momentum``，默认值为None，在``O2``模式下使用master weight策略。
    - **save_dtype** (str|None, 可选) - 网络存储类型，可为float16、float32、float64。通过 ``save_dtype`` 可指定通过 ``paddle.save`` 和 ``paddle.jit.save`` 存储的网络参数数据类型。默认为None，采用现有网络参数类型进行存储。


**代码示例**：
COPY-FROM: paddle.amp.decorate
