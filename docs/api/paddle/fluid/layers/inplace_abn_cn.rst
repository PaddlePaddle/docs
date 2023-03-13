.. _cn_api_fluid_layers_inplace_abn:

inplace_abn
-------------------------------

**注意：该 API 仅支持【静态图】模式**

.. py:function:: paddle.fluid.layers.inplace_abn(input, act=None, is_test=False, momentum=0.9, epsilon=1e-05, param_attr=None, bias_attr=None, data_layout='NCHW', name=None, moving_mean_name=None, moving_variance_name=None, do_model_average_for_mean_and_var=False, use_global_stats=False, act_alpha=1.0)

就地批正则化化激活层（Inplace Activation Batch Normalization Layer）

此层使用就地内存计算批处理正则化和激活来实现节省内存，有关批量正则化计算，请参见 ``fluid.layers.batch_norm``，有关就地激活批正则化化的计算，请参考 `In-Place Activated BatchNorm for Memory-Optimized Training of DNNs <https://arxiv.org/abs/1712.02616>`_ 。

参数
::::::::::::

    - **input** (Variable) - inplace_abn 算子的输入特征，是一个 Variable 类型，输入维度可以是 2, 3, 4, 5。数据类型：flaot16, float32, float64。
    - **act** （string）- 激活函数类型，可以是 leaky_realu、relu、prelu 等。默认：None。
    - **is_test** （bool） - 指示它是否在测试阶段，非训练阶段使用训练过程中统计到的全局均值和全局方差。默认：False。
    - **momentum** （float|Variable）- 此值用于计算 moving_mean 和 moving_var，是一个 float 类型或者一个 shape 为[1]，数据类型为 float32 的 Variable 类型。更新公式为：:math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)` ， :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)`，默认：0.9。
    - **epsilon** （float）- 加在分母上为了数值稳定的值。默认：1e-5。
    - **param_attr** (ParamAttr|None)：指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。inplace_abn 算子默认的权重初始化是 1.0。
    - **bias_attr** （ParamAttr|None）- 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。inplace_abn 算子默认的偏置初始化是 0.0。
    - **data_layout** （string) - 指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N 是批尺寸，C 是通道数，H 是特征高度，W 是特征宽度。默认值："NCHW"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **moving_mean_name** （string）- moving_mean 的名称，存储全局均值。如果将其设置为 None, ``inplace_abn`` 将随机命名全局均值；否则，``inplace_abn`` 将命名全局均值为 ``moving_mean_name``。默认：None。
    - **moving_variance_name** （string）- moving_variance 的名称，存储全局变量。如果将其设置为 None, ``inplace_abn`` 将随机命名全局方差；否则，``inplace_abn`` 将命名全局方差为 ``moving_variance_name``。默认：None。
    - **do_model_average_for_mean_and_var** （bool，默认 False）- 是否为 mean 和 variance 做模型均值。
    - **use_global_stats** （bool） – 是否使用全局均值和方差。在预测或测试模式下，将 use_global_stats 设置为 true 或将 is_test 设置为 true，并且行为是等效的。在训练模式中，当设置 use_global_stats 为 True 时，在训练期间也使用全局均值和方差。默认：False。
    - **act_alpha** （float） – 当 ``act`` 参数为 None、leaky-relu、elu 时，会使用就地批正则化激活算法，可通过此参数给定 leaky-relu、elu 的 ``alpha`` 值。默认：1.0。


返回
::::::::::::
 维度和输入相同的 Tensor，在输入中运用批正则后的结果。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.inplace_abn
