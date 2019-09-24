.. _cn_api_fluid_layers_clip_by_norm:

clip_by_norm
-------------------------------

.. py:function:: paddle.fluid.layers.clip_by_norm(x, max_norm, name=None)

ClipByNorm算子

此算子将输入 ``X`` 的L2范数限制在 ``max_norm`` 内。如果 ``X`` 的L2范数小于或等于 ``max_norm``  ，则输出（Out）将与 ``X`` 相同。如果X的L2范数大于 ``max_norm`` ，则 ``X`` 将被线性缩放，使得输出（Out）的L2范数等于 ``max_norm`` ，如下面的公式
所示：

.. math::
         Out = \frac{max\_norm * X}{norm(X)}

其中， :math:`norm（X）` 代表 ``x`` 的L2范数。


参数：
        - **x** (Variable)- 多维Tensor或LoDTensor，数据类型为float32。clip_by_norm运算的输入，维数必须在[1,9]之间。
        - **max_norm** (float32)- 最大范数值。
        - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：        表示为输出Tensor或LoDTensor，数据类型为float32。和输入(X)具有相同的形状.


返回类型：Variable


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(
        name='data', shape=[1], dtype='float32')
    reward = fluid.layers.clip_by_norm(x=input, max_norm=1.0)