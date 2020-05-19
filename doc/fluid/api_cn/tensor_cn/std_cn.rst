std
-------------------------------

.. py:function:: paddle.std(input, axis=None, keepdim=False, unbiased=True, out=None, name=None)

:alias_main: paddle.std
:alias: paddle.std,paddle.tensor.std,paddle.tensor.stat.std



根据指定的axis计算input的标准差。

参数：
    - **input** (Variable) - 数据类型为float32,float64。要计算标准差的输入。
    - **axis** (list|int, 可选) - 根据axis计算标准差。如果设为 `None`，计算 :attr:`input` 的所有元素的标准差并返回shape为[1]的单个结果。如果不是 `None`, 则设定的axis的值必须在 :math:`[-rank(input), rank(input))` 范围内。 如果 :math:`axis[i] < 0`, 则要计算的axis是 :math:`rank(input) + axis[i]`。
    - **keepdim** (bool, 可选) - 是否在输出中保留减小的维度。如 keep_dim 为False，输出张量的维度将比输入张量小， 为True时则维度相同。默认值：False。
    - **unbiased** (bool, 可选) - 是否使用无偏估计来计算标准差。使用 :math:`N` 来代表在axis上的维度，如果 unbiased 为True, 则在计算中使用 :math:`N - 1` 作为除数。为False时将使用 :math:`N` 作为除数。默认值：True。
    - **out** (Variable, 可选) - 如果out不是None, 则将标准差结果存储在out中。默认值：None。
    - **name** (str，可选) – 具体用法请参见 :ref:`cn_api_guide_Name` ，一般无需设置，默认值：None。


返回: 计算出来的标准差。

返回类型: Variable（Tensor) ，数据类型和 :attr:`input` 相同。如果 :attr:`out = None`, 返回包含标准差的新Variable , 否则的话返回 :attr:`out` 的引用。

**代码示例**

.. code-block:: python
    
    import paddle
    import paddle.fluid as fluid
    # x is a Tensor variable with following elements:
    #    [[0.2, 0.3, 0.5, 0.9]
    #     [0.1, 0.2, 0.6, 0.7]]
    # Each example is followed by the corresponding output tensor.
    x = fluid.data(name='x', shape=[2, 4], dtype='float32')
    paddle.std(x)  # [0.28252685] 
    paddle.std(x, axis=[0])  # [0.0707107, 0.07071075, 0.07071064, 0.1414217]
    paddle.std(x, axis=[-1])  # [0.30956957, 0.29439208]