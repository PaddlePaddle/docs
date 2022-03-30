.. _cn_api_paddle_cn_diag:

diag
-------------------------------

.. py:function:: paddle.diag(x, offset=0, padding_value=0, name=None)


如果 ``x`` 是向量（1-D张量），则返回带有 ``x`` 元素作为对角线的2-D方阵。

如果 ``x`` 是矩阵（2-D张量），则提取 ``x`` 的对角线元素，以1-D张量返回。

参数 ``offset`` 控制对角线偏移量：

- 如果 ``offset`` = 0，则为主对角线。
- 如果 ``offset`` > 0，则为上对角线。
- 如果 ``offset`` < 0，则为下对角线。

参数
:::::::::
    - x（Tensor）：输入的 `Tensor`。它的形状可以是一维或二维。其数据类型应为float32，float64，int32，int64。
    - offset（int，可选）：对角线偏移量。正值表示上对角线，0表示主对角线，负值表示下对角线。
    - padding_value（int|float，可选）：使用此值来填充指定对角线以外的区域。仅在输入为一维张量时生效。默认值为0。
    - name (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
``Tensor``，方阵或向量。输出数据类型与输入数据类型相同。


代码示例 1
:::::::::

.. code-block:: python

        import paddle

        x = paddle.to_tensor([1, 2, 3])
        y = paddle.diag(x)
        print(y)
        # [[1 0 0]
        #  [0 2 0]
        #  [0 0 3]]

        y = paddle.diag(x, offset=1)
        print(y)
        # [[0 1 0 0]
        #  [0 0 2 0]
        #  [0 0 0 3]
        #  [0 0 0 0]]

        y = paddle.diag(x, padding_value=6)
        print(y)
        # [[1 6 6]
        #  [6 2 6]
        #  [6 6 3]]


代码示例 2
:::::::::

.. code-block:: python

        import paddle

        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
        y = paddle.diag(x)
        print(y)
        # [1 5]

        y = paddle.diag(x, offset=1)
        print(y)
        # [2 6]

        y = paddle.diag(x, offset=-1)
        print(y)
        # [4]





