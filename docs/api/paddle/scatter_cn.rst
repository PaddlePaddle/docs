.. _cn_api_paddle_cn_scatter:

scatter
-------------------------------
.. py:function:: paddle.scatter(x, index, updates, overwrite=True, name=None)


通过基于 ``updates`` 来更新选定索引 ``index`` 上的输入来获得输出。具体行为如下：

    .. code-block:: python

        import numpy as np
        #input:
        x = np.array([[1, 1], [2, 2], [3, 3]])
        index = np.array([2, 1, 0, 1])
        # shape of updates should be the same as x
        # shape of updates with dim > 1 should be the same as input
        updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        overwrite = False
        # calculation:
        if not overwrite:
            for i in range(len(index)):
                x[index[i]] = np.zeros((2))
        for i in range(len(index)):
            if (overwrite):
                x[index[i]] = updates[i]
            else:
                x[index[i]] += updates[i]
        # output:
        out = np.array([[3, 3], [6, 6], [1, 1]])
        out.shape # [3, 2]

**Notice：**
因为 ``updates`` 的应用顺序是不确定的，因此，如果索引 ``index`` 包含重复项，则输出将具有不确定性。


参数
:::::::::
    - **x** (Tensor) - ndim> = 1 的输入 N-D 张量。数据类型可以是 float32，float64。
    - **index** （Tensor）- 一维 Tensor。数据类型可以是 int32，int64。 ``index`` 的长度不能超过 ``updates`` 的长度，并且 ``index`` 中的值不能超过输入的长度。
    - **updates** （Tensor）- 根据 ``index`` 使用 ``update`` 参数更新输入 ``x``。形状应与输入 ``x`` 相同，并且 dim>1 的 dim 值应与输入 ``x`` 相同。
    - **overwrite** （bool，可选）- 指定索引 ``index`` 相同时，更新输出的方式。如果为 True，则使用覆盖模式更新相同索引的输出，如果为 False，则使用累加模式更新相同索引的输出。默认值为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，与 x 有相同形状和数据类型。


代码示例
:::::::::

.. code-block:: python

        import paddle
        import numpy as np

        x_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
        index_data = np.array([2, 1, 0, 1]).astype(np.int64)
        updates_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float32)

        x = paddle.to_tensor(x_data)
        index = paddle.to_tensor(index_data)
        updates = paddle.to_tensor(updates_data)

        output1 = paddle.scatter(x, index, updates, overwrite=False)
        # [[3., 3.],
        #  [6., 6.],
        #  [1., 1.]]
        output2 = paddle.scatter(x, index, updates, overwrite=True)
        # CPU device:
        # [[3., 3.],
        #  [4., 4.],
        #  [1., 1.]]
        # GPU device maybe have two results because of the repeated numbers in index
        # result 1:
        # [[3., 3.],
        #  [4., 4.],
        #  [1., 1.]]
        # result 2:
        # [[3., 3.],
        #  [2., 2.],
        #  [1., 1.]]
