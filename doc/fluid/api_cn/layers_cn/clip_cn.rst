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
        - **x** (Tensor|LoDTensor)- 数据类型为float的Tensor或者LoDTensor，clip_by_norm运算的输入，维数必须在[1,9]之间。
        - **max_norm** (float)- 最大范数值。
        - **name** (str|None)- 输出变量的名称。

返回：        Variable，数据类型为float的Tensor或者LoDTensor。操作后的输出和输入(X)具有相同的形状.

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(
        name='data', shape=[1], dtype='float32')
    reward = fluid.layers.clip_by_norm(x=input, max_norm=1.0)