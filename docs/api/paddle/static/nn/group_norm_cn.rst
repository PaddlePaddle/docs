.. _cn_api_fluid_layers_group_norm:

group_norm
-------------------------------


.. py:function::  paddle.static.nn.group_norm(input, groups, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, data_layout='NCHW', name=None)

论文参考：`Group Normalization <https://arxiv.org/abs/1803.08494>`_

参数
:::::::::

  - **input** (Tensor)：维度大于 1 的 Tensor，数据类型为 float32 或 float64。
  - **groups** (int)：从 channel 中分离出来的 group 的数目，数据类型为 int32。
  - **epsilon** (float，可选)：为防止方差除以零，增加一个很小的值。数据类型为 float32。默认值：1e-05。
  - **param_attr** (ParamAttr|bool，可选)：指定权重参数属性的对象。若 ``param_attr`` 为 bool 类型，只支持为 False，表示没有权重参数。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr|bool，可选)：指定偏置参数属性的对象。若 ``bias_attr`` 为 bool 类型，只支持为 False，表示没有偏置参数。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **act** (str，可选)：将激活应用于输出的 group normalization。
  - **data_layout** (str，可选)：指定输入的数据格式，输出的数据格式将与输入保持一致，可以是"NCHW"和"NHWC"，默认值："NCHW"。如果是"NCHW"，则数据按[批大小，输入通道数，* ]的顺序存储。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor，数据类型和格式与 `input` 一致。

代码示例
:::::::::

COPY-FROM: paddle.static.nn.group_norm
