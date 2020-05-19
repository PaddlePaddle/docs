.. _cn_api_fluid_layers_group_norm:

group_norm
-------------------------------


.. py:function::  paddle.fluid.layers.group_norm(input, groups, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, data_layout='NCHW', name=None)

:api_attr: 声明式编程模式（静态图)



参考论文： `Group Normalization <https://arxiv.org/abs/1803.08494>`_

参数：
  - **input** (Variable)：输入为4-D Tensor，数据类型为float32或float64。
  - **groups** (int)：从 channel 中分离出来的 group 的数目，数据类型为int32。
  - **epsilon** (float，可选)：为防止方差除以零，增加一个很小的值。数据类型为float32。默认值：1e-05。
  - **param_attr** (ParamAttr|bool，可选) ：指定权重参数属性的对象。若 ``param_attr`` 为bool类型，只支持为False，表示没有权重参数。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr|bool，可选) : 指定偏置参数属性的对象。若 ``bias_attr`` 为bool类型，只支持为False，表示没有偏置参数。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **act** (str，可选)：将激活应用于输出的 group normalizaiton。
  - **data_layout** (str，可选)：指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"。N是批尺寸，C是通道数，H是特征高度，W是特征宽度。默认值："NCHW"。
  - **name** (str，可选)：具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：4-D Tensor，数据类型和格式与 `input` 一致。

返回类型：Variable

抛出异常：
    - ``ValueError`` - 如果 ``data_layout`` 既不是"NCHW"也不是"NHWC"。
    - ``ValueError`` - 如果 ``groups`` 小于1，或者 ``groups`` 大于输入的通道数。
    - ``ShapeError`` - 如果  ``param_attr`` (Scale) 或者 ``bias_attr`` (Bias) 不是 1-D Tensor。
    - ``ShapeError`` - 如果  ``param_attr`` (Scale) 或者 ``bias_attr`` (Bias) 的大小与输入的通道数不相等。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.data(name='data', shape=[None, 8, 32, 32], dtype='float32')
    x = fluid.layers.group_norm(input=data, groups=4)










