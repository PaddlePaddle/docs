.. _cn_api_paddle_diagflat:

diagflat
-------------------------------

.. py:function:: paddle.diagflat(x, offset=0, name=None)


如果 ``x`` 是一维张量，则返回带有 ``x`` 元素作为对角线的二维方阵。

如果 ``x`` 是大于等于二维的张量，则返回一个二维方阵，其对角线元素为 ``x`` 在连续维度展开得到的一维张量的元素。

参数 ``offset`` 控制对角线偏移量：

- 如果 ``offset`` = 0，则为主对角线。
- 如果 ``offset`` > 0，则为上对角线。
- 如果 ``offset`` < 0，则为下对角线。

参数
:::::::::
    - x（Tensor）：输入的 `Tensor`。它的形状可以是任意维度。其数据类型应为float32，float64，int32，int64。
    - offset（int，可选）：对角线偏移量。正值表示上对角线，0表示主对角线，负值表示下对角线。
    - name (str，可选）：操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor``，方阵。输出数据类型与输入数据类型相同。


代码示例 1
:::::::::

.. code-block:: python

        import paddle

        x = paddle.to_tensor([1, 2, 3])
        y = paddle.diagflat(x)
        print(y.numpy())
        # [[1 0 0]
        #  [0 2 0]
        #  [0 0 3]]

        y = paddle.diagflat(x, offset=1)
        print(y.numpy())
        # [[0 1 0 0]
        #  [0 0 2 0]
        #  [0 0 0 3]
        #  [0 0 0 0]]

        y = paddle.diagflat(x, offset=-1)
        print(y.numpy())
        # [[0 0 0 0]
        #  [1 0 0 0]
        #  [0 2 0 0]
        #  [0 0 3 0]]


代码示例 2
:::::::::

.. code-block:: python

        import paddle

        x = paddle.to_tensor([[1, 2], [3, 4]])
        y = paddle.diagflat(x)
        print(y.numpy())
        # [[1 0 0 0]
        #  [0 2 0 0]
        #  [0 0 3 0]
        #  [0 0 0 4]]

        y = paddle.diagflat(x, offset=1)
        print(y.numpy())
        # [[0 1 0 0 0]
        #  [0 0 2 0 0]
        #  [0 0 0 3 0]
        #  [0 0 0 0 4]
        #  [0 0 0 0 0]]

        y = paddle.diagflat(x, offset=-1)
        print(y.numpy())
        # [[0 0 0 0 0]
        #  [1 0 0 0 0]
        #  [0 2 0 0 0]
        #  [0 0 3 0 0]
        #  [0 0 0 4 0]]





