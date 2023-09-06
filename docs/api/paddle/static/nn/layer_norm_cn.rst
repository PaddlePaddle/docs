.. _cn_api_fluid_layers_layer_norm:

layer_norm
-------------------------------


.. py:function:: paddle.static.nn.layer_norm(input, scale=True, shift=True, begin_norm_axis=1, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, name=None)




实现了层归一化层（Layer Normalization Layer），其可以应用于小批量输入数据。

论文参考：`Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

计算公式如下

.. math::
            \\\mu=\frac{1}{H}\sum_{i=1}^{H}x_i\\

            \\\sigma=\sqrt{\frac{1}{H}\sum_i^H{(x_i-\mu)^2} + \epsilon}\\

             \\y=f(\frac{g}{\sigma}(x-\mu) + b)\\

- :math:`x`：该层神经元的向量表示；
- :math:`H`：层中隐藏神经元个数；
- :math:`\epsilon`：添加较小的值到方差中以防止除零；
- :math:`g`：可训练的比例参数；
- :math:`b`：可训练的偏差参数。

参数
::::::::::::

  - **input** (Tensor) - 维度为任意维度的多维 ``Tensor``，数据类型为 float32 或 float64。
  - **scale** (bool，可选) - 指明是否在归一化后学习自适应增益 ``g``。默认值：True。
  - **shift** (bool，可选) - 指明是否在归一化后学习自适应偏差 ``b``。默认值：True。
  - **begin_norm_axis** (int，可选) - 指明归一化将沿着 ``begin_norm_axis`` 到 ``rank（input）`` 的维度执行。默认值：1。
  - **epsilon** (float，可选) - 指明在计算过程中是否添加较小的值到方差中以防止除零。默认值：1e-05。
  - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **act** (str，可选) - 应用于输出上的激活函数，如 tanh、softmax、sigmoid，relu 等，支持列表请参考 :ref:`api_guide_activations`，默认值为 None。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
表示归一化结果的 ``Tensor``，数据类型和 ``input`` 一致，返回维度和 ``input`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.static.nn.layer_norm
