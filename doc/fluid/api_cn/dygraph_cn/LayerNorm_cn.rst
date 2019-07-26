.. _cn_api_fluid_dygraph_LayerNorm:

LayerNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.LayerNorm(name_scope, scale=True, shift=True, begin_norm_axis=1, epsilon=1e-05, param_attr=None, bias_attr=None, act=None)


假设特征向量存在于维度 ``begin_norm_axis ... rank (input）`` 上，计算大小为 ``H`` 的特征向量a在该维度上的矩统计量，然后使用相应的统计量对每个特征向量进行归一化。 之后，如果设置了 ``scale`` 和 ``shift`` ，则在标准化的张量上应用可学习的增益和偏差以进行缩放和移位。

请参考 `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

公式如下

.. math::
            \\\mu=\frac{1}{H}\sum_{i=1}^{H}a_i\\
.. math::
            \\\sigma=\sqrt{\frac{1}{H}\sum_i^H{(a_i-\mu)^2}}\\
.. math::
             \\h=f(\frac{g}{\sigma}(a-\mu) + b)\\

- :math:`\alpha` : 该层神经元输入总和的向量表示
- :math:`H` : 层中隐藏的神经元个数
- :math:`g` : 可训练的缩放因子参数
- :math:`b` : 可训练的bias参数


参数:
    - **name_scope** (str) – 该类的名称
    - **scale** （bool） - 是否在归一化后学习自适应增益g。默认为True。
    - **shift** （bool） - 是否在归一化后学习自适应偏差b。默认为True。
    - **begin_norm_axis** （int） - ``begin_norm_axis`` 到 ``rank（input）`` 的维度执行规范化。默认1。
    - **epsilon** （float） - 添加到方差的很小的值，以防止除零。默认1e-05。
    - **param_attr** （ParamAttr | None） - 可学习增益g的参数属性。如果  ``scale`` 为False，则省略 ``param_attr`` 。如果 ``scale`` 为True且 ``param_attr`` 为None，则默认 ``ParamAttr`` 将作为比例。如果添加了 ``param_attr``， 则将其初始化为1。默认None。
    - **bias_attr** （ParamAttr | None） - 可学习偏差的参数属性b。如果 ``shift`` 为False，则省略 ``bias_attr`` 。如果 ``shift`` 为True且 ``param_attr`` 为None，则默认 ``ParamAttr`` 将作为偏差。如果添加了 ``bias_attr`` ，则将其初始化为0。默认None。
    - **act** （str） - 激活函数。默认 None


返回： 标准化后的结果

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy

    with fluid.dygraph.guard():
        x = numpy.random.random((3, 32, 32)).astype('float32')
        layerNorm = fluid.dygraph.nn.LayerNorm(
              'LayerNorm', begin_norm_axis=1)
       ret = layerNorm(fluid.dygraph.base.to_variable(x))





