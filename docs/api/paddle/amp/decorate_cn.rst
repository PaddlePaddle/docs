.. _cn_api_amp_decorate:

decorate
-------------------------------

.. py:function:: paddle.amp.decorate(models, optimizers=None, level='O1', master_weight=None, save_dtype=None)


装饰神经网络参数，来支持动态图模式下执行的算子的自动混合精度策略（AMP）。
在``O1``模式下，该函数不做任何处理，直接返回输入的models和optimizers。在``O2``模式下，将对输入的网络参数数据类型由float32转为float16，（除BatchNorm和LayerNorm）。
通过该函数可为支持master weight策略的优化器开启master weight策略，以保证训练精度。通过 ``save_dtype`` 可指定 ``paddle.save`` 和 ``paddle.jit.save`` 存储的网络参数数据类型。


参数：
:::::::::
    - **models** (Layer|list of Layer) - 网络模型。在``O2``模式下，输入的模型参数将由float32转为float16。
    - **optimizers** (Optimizer|list of Optimizer, 可选) - 优化器，默认值为None，若传入优化器或由优化器组成的list列表，将依据master_weight对优化器的master_weight属性进行设置。
    - **level** (str, 可选) - 混合精度训练模式，默认``O1``模式。
    - **master_weight** (bool|None, 可选) - 是否使用master weight策略。支持maser weight策略的优化器包括``adam``、``adamW``、``momentum``，默认值为None，在``O2``模式下使用master weight策略。
    - **save_dtype** (str|None, 可选) - 网络存储类型，可为float16、float32、float64。通过 ``save_dtype`` 可指定通过 ``paddle.save`` 和 ``paddle.jit.save`` 存储的网络参数数据类型。默认为None，采用现有网络参数类型进行存储。


代码示例：
:::::::::
COPY-FROM: paddle.amp.decorate
