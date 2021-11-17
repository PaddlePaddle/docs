.. _cn_api_incubate_send_recv:

send_recv
-------------------------------

.. py:function:: paddle.incubate.send_recv(x, src_index, dst_index, pool_type="sum", name=None)

此API主要应用于图学习领域，目的是为了减少在消息传递过程中带来的中间变量显存或内存的损耗。其中， ``x`` 作为输入Tensor，首先利用 ``src_index`` 作为索引来gather出在 ``x`` 中相应位置的数据，随后再将gather出的结果利用 ``dst_index`` 来scatter到对应的输出结果中，其中 ``pool_type`` 表示scatter的不同处理方式，包括sum、mean、max、min共计4种处理模式。

.. code-block:: text

        X = [[0, 2, 3],
             [1, 4, 5],
             [2, 6, 7]]

        src_index = [0, 1, 2, 0]

        dst_index = [1, 2, 1, 0]

        pool_type = "sum"

        Then:

        Out = [[0, 2, 3],
               [2, 8, 10],
               [1, 4, 5]]

参数
:::::::::
    - x (Tensor) - 输入的 Tensor，数据类型为：float32、float64、int32、int64。
    - src_index (Tensor) - 1-D Tensor，数据类型为：int32、int64。
    - dst_index (Tensor) - 1-D Tensor，数据类型为：int32、int64。注意： ``dst_index`` 的形状应当与 ``src_index`` 一致。
    - pool_type (str) - scatter结果的不同处理方式，包括sum、mean、max、min。 默认值为 sum。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
``Tensor`` ，维度和数据类型都与 ``x`` 相同，存储运算后的结果。


代码示例
::::::::::

.. code-block:: python

    import numpy as np
    import paddle

    x = paddle.to_tensor(np.array([[0, 2, 3], [1, 4, 5], [2, 6, 7]]), dtype="float32")
    indexes = paddle.to_tensor(np.array([[0, 1], [1, 2], [2, 1], [0, 0]]), dtype="int32")
    src_index = indexes[:, 0]
    dst_index = indexes[:, 1]
    out = paddle.incubate.send_recv(x, src_index, dst_index, pool_type="sum")
    # Outputs: [[0., 2., 3.], [2., 8., 10.], [1., 4., 5.]]
