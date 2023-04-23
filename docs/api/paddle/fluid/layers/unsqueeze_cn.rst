.. _cn_api_fluid_layers_unsqueeze:

unsqueeze
-------------------------------

.. py:function:: paddle.fluid.layers.unsqueeze(input, axes, name=None)




该OP向输入（input）的shape中一个或多个位置（axes）插入维度。

- 示例：

.. code-block:: python

    输入：
      X.shape = [2, 3]
      X.data = [[1, 2, 3], 
                [4，5，6]]
      axes = [0, 2]
    输出（在X的第0维和第2维插入新维度）：
      Out.shape = [1, 2, 1, 3]
      Out.data = [[[[1, 2, 3]],
                    [[4, 5, 6]]]]
      
参数
::::::::::::

    - **input** (Variable)- 多维 ``Tensor``，数据类型为 ``float32``， ``float64``， ``int8``， ``int32``，或 ``int64``。
    - **axes** (int|list|tuple|Variable) - 表示要插入维度的位置。数据类型是 ``int32``。如果 ``axes`` 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 ``Tensor``。如果 ``axes`` 的类型是 ``Variable``，则是1-D ``Tensor``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
扩展维度后的多维Tensor

返回类型
::::::::::::
Variable

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.data(name='x', shape=[5, 10])
    y = fluid.layers.unsqueeze(input=x, axes=[1])
    # y.shape is [5, 1, 10]
