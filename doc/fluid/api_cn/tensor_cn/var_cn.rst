var
-------------------------------

.. py:function:: paddle.var(input, axis=None, keepdim=False, unbiased=True, out=None, name=None)

:alias_main: paddle.var
:alias: paddle.var,paddle.tensor.var,paddle.tensor.stat.var






 沿给定的轴 axis 计算输入变量所有元素的方差。

  参数：
     - **input** (Variable) - 要计算方差的输入变量，支持的数据类型为 float32 或 float64。
     - **axis** (list|int, 可选) - 给定的轴。如果设为 `None`，计算 :attr:`input` 的所有元素的方差并返回形状为 [1] 的单个结果。如果非 `None`, 则给定的 axis 的值必须在 :math:`[-rank(input), rank(input))` 范围内。 如果 :math:`axis[i] < 0`, 则实际的 axis 是 :math:`rank(input) + axis[i]`。
     - **keepdim** (bool, 可选) - 是否在输出中保留被规约的维度。如 keep_dim 为False，输出张量的维度将比输入张量小， 为True时则维度相同。默认值：False。
     - **unbiased** (bool, 可选) - 是否使用无偏估计来计算方差。使用 :math:`N` 来代表在 axis 上的维度，如果 unbiased 为True, 则在计算中使用 :math:`N - 1` 作为除数。为 False 时将使用 :math:`N` 作为除数。默认值：True。
     - **out** (Variable, 可选) - 如果out不是None, 则将方差结果存储在 out 中。默认值：None。
     - **name** (str，可选) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值：None。

 
  返回: 计算出来的方差。

  返回类型: Variable（Tensor) ，数据类型和 :attr:`input` 相同。如果 :attr:`out = None`, 返回包含方差的新 Variable , 否则返回 :attr:`out` 的引用。

  **代码示例**

  .. code-block:: python
    
        import numpy as np
        import paddle
        import paddle.fluid.dygraph as dg
        a = np.array([[1.0, 2.0], [3.0, 4.0]]).astype("float32")
        with dg.guard():
            data = dg.to_variable(a)
            variance = paddle.var(data, axis=[1])
            print(variance.numpy())   
            # [0.5 0.5] 
