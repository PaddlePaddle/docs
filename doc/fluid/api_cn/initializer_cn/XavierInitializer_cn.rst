.. _cn_api_fluid_initializer_XavierInitializer:

XavierInitializer
-------------------------------

.. py:class:: paddle.fluid.initializer.XavierInitializer(uniform=True, fan_in=None, fan_out=None, seed=0)

该类实现Xavier权重初始化方法（ Xavier weight initializer），Xavier权重初始化方法出自Xavier Glorot和Yoshua Bengio的论文 `Understanding the difficulty of training deep feedforward neural networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

该初始化函数用于保持所有层的梯度尺度几乎一致。

在均匀分布的情况下，取值范围为[-x,x]，其中：

.. math::

    x = \sqrt{\frac{6.0}{fan\_in+fan\_out}}

正态分布的情况下，均值为0，标准差为：

.. math::
    
    x = \sqrt{\frac{2.0}{fan\_in+fan\_out}}

参数：
    - **uniform** (bool) - 是否用均匀分布，默认为True。如果为False，则使用正态分布。
    - **fan_in** (float) - 当前网络层的输入神经元个数。如果为None，则从变量中推断。
    - **fan_out** (float) - 当前网络层的输出神经元个数。如果为None，则从变量中推断。
    - **seed** (int) - 随机种子

.. note::

    在大多数情况下推荐将fan_in和fan_out设置为None

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    queries = fluid.layers.data(name='x', shape=[1], dtype='float32')
    fc = fluid.layers.fc(
        input=queries, size=10,
        param_attr=fluid.initializer.Xavier(uniform=False))






