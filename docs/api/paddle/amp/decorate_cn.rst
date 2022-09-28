.. _cn_api_amp_decorate:

decorate
-------------------------------

.. py:function:: paddle.amp.decorate(models, optimizers=None, level='O1', dtype='float16', master_weight=None, save_dtype=None)


装饰神经网络参数，来支持动态图模式下执行的算子的自动混合精度策略（AMP）。
在``O1``模式下，该函数不做任何处理，直接返回输入的 models 和 optimizers。在``O2``模式下，将对输入的网络参数数据类型由 float32 转为 float16 或 bfloat16，（除 BatchNorm 和 LayerNorm）。
通过该函数可为支持 master weight 策略的优化器开启 master weight 策略，以保证训练精度。通过 ``save_dtype`` 可指定 ``paddle.save`` 和 ``paddle.jit.save`` 存储的网络参数数据类型。


参数
::::::::::::

    - **models** (Layer|list of Layer) - 网络模型。在``O2``模式下，输入的模型参数将由 float32 转为 float16 或 bfloat16。
    - **optimizers** (Optimizer|list of Optimizer，可选) - 优化器，默认值为 None，若传入优化器或由优化器组成的 list 列表，将依据 master_weight 对优化器的 master_weight 属性进行设置。
    - **level** (str，可选) - 混合精度训练模式，默认``O1``模式。
    - **dtype** (str，可选) - 混合精度训练数据类型使用 float16 还是 bfloat16，默认为 float16 类型。
    - **master_weight** (bool|None，可选) - 是否使用 master weight 策略。支持 master weight 策略的优化器包括``adam``、``adamW``、``momentum``，默认值为 None，在``O2``模式下使用 master weight 策略。
    - **save_dtype** (str|None，可选) - 网络存储类型，可为 float16、bfloat16、float32、float64。通过 ``save_dtype`` 可指定通过 ``paddle.save`` 和 ``paddle.jit.save`` 存储的网络参数数据类型。默认为 None，采用现有网络参数类型进行存储。


代码示例
:::::::::
COPY-FROM: paddle.amp.decorate
