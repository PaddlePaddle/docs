.. _cn_api_fluid_dygraph_LayerNorm:

LayerNorm
-------------------------------

.. py:class:: paddle.fluid.dygraph.LayerNorm(name_scope, scale=True, shift=True, begin_norm_axis=1, epsilon=1e-05, param_attr=None, bias_attr=None, act=None)

该OP实现了层正则化层（Layer Normalization Layer），其可以应用于小批量输入数据。更多详情请参考：`Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

计算公式如下

.. math::
            \\\mu=\frac{1}{H}\sum_{i=1}^{H}x_i\\

            \\\sigma=\sqrt{\frac{1}{H}\sum_i^H{(x_i-\mu)^2} + \epsilon}\\

             \\y=f(\frac{g}{\sigma}(x-\mu) + b)\\

- :math:`x` : 该层神经元的向量表示
- :math:`H` : 层中隐藏神经元个数
- :math:`\epsilon` : 添加较小的值到方差中以防止除零
- :math:`g` : 可训练的比例参数
- :math:`b` : 可训练的偏差参数


参数：
    - **name_scope** (str) – 该类的名称。
    - **scale** (bool, 可选) - 指明是否在归一化后学习自适应增益 ``g`` 。默认值：True。
    - **shift** (bool, 可选) - 指明是否在归一化后学习自适应偏差 ``b`` 。默认值：True。
    - **begin_norm_axis** (int, 可选) - 指明归一化将沿着 ``begin_norm_axis`` 到 ``rank（input）`` 的维度执行。默认值：1。
    - **epsilon** (float, 可选) - 指明在计算过程中是否添加较小的值到方差中以防止除零。默认值：1e-05。
    - **param_attr** (ParamAttr, 可选) - 指明可学习参数 ``g`` 的参数属性。如果 ``scale`` 为False，则省略 ``param_attr`` 。如果 ``scale`` 为True且 ``param_attr`` 为None，则以默认的方式生成 ``ParamAttr`` 对象，将参数初始化为1。默认值：None。
    - **bias_attr** (ParamAttr, 可选) - 指明可学习参数 ``b`` 的参数属性。如果 ``shift`` 为False，则省略 ``bias_attr`` 。如果 ``shift`` 为True且 ``param_attr`` 为None，则以默认的方式生成 ``ParamAttr`` 对象，将参数初始化为0。默认值：None。
    - **act** (str, 可选) - 指明激活函数类型。默认值：None。


返回：无

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph.base import to_variable
    import numpy

    x = numpy.random.random((3, 32, 32)).astype('float32')
    with fluid.dygraph.guard():
        x = to_variable(x)
        layernorm = fluid.LayerNorm('LayerNorm', begin_norm_axis=1)
        ret = layernorm(x)


