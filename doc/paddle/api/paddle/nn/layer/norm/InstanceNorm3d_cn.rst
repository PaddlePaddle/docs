.. _cn_api_nn_cn_InstanceNorm3d:

InstanceNorm3d
-------------------------------

.. py:class:: paddle.nn.InstanceNorm3d(num_features, epsilon=1e-05, momentum=0.9, weight_attr=None, bias_attr=None,  track_running_stats=True, data_format="NCDHW", name=None):

该接口用于构建 ``InstanceNorm3d`` 类的一个可调用对象，具体用法参照 ``代码示例`` 。可以处理5D的Tensor, 实现了实例归一化层（Instance Normalization Layer）的功能。更多详情请参考 : Instance Normalization: The Missing Ingredient for Fast Stylization .

``input`` 是mini-batch的输入。

.. math::
    \mu_{\beta}        &\gets \frac{1}{m} \sum_{i=1}^{m} x_i                                 \quad &// mean  \\
    \sigma_{\beta}^{2} &\gets \frac{1}{m} \sum_{i=1}^{m}(x_i - \mu_{\beta})^2               \quad &// variance \\
    \hat{x_i}          &\gets \frac{x_i - \mu_\beta} {\sqrt{\sigma_{\beta}^{2} + \epsilon}}  \quad &// normalize \\
    y_i &\gets \gamma \hat{x_i} + \beta                                                      \quad &// scale-and-shift

Note:
    `H` 是高度, `W` 是宽度.


参数：
    - **num_features** (int) - 指明输入 ``Tensor`` 的通道数量。
    - **epsilon** (float, 可选) - 为了数值稳定加在分母上的值。默认值：1e-05。
    - **momentum** (float, 可选) - 此值用于计算 ``moving_mean`` 和 ``moving_var`` 。默认值：0.9。更新公式如上所示。
    - **weight_attr** (ParamAttr|bool, 可选) - 指定权重参数属性的对象。如果为False, 则表示每个通道的伸缩固定为1，不可改变。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_ParamAttr` 。
    - **bias_attr** (ParamAttr, 可选) - 指定偏置参数属性的对象。如果为False, 则表示每一个通道的偏移固定为0，不可改变。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_ParamAttr` 。
    - **track_running_stats** (bool, 可选) – 指示是否使用全局均值和方差。在训练时，设置为True表示在训练期间将保存全局均值和方差用于推理。推理时此属性只能设置为True。默认值：True。
    - **data_format** (string, 可选) - 指定输入数据格式，数据格式可以为"NCDHW"。默认值：“NCDHW”。
    - **name** (string, 可选) – InstanceNorm的名称, 默认值为None。更多信息请参见 :ref:`api_guide_Name` 。


返回：无

形状：
    - input: 形状为5-D Tensor。
    - output: 和输入形状一样。

.. note::
目前设置track_running_stats和momentum是无效的。之后的版本会修复此问题。
    

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.disable_static()
    np.random.seed(123)
    x_data = np.random.random(size=(2, 2, 2, 2, 3)).astype('float32')
    x = paddle.to_tensor(x_data) 
    instance_norm = paddle.nn.InstanceNorm3d(2)
    instance_norm_out = instance_norm(x)

    print(instance_norm_out.numpy())

