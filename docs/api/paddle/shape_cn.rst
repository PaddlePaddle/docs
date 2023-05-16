.. _cn_api_fluid_layers_shape:

shape
-------------------------------

.. py:function:: paddle.shape(input)




shape 层。

获得输入 Tensor 或 SelectedRows 的 shape。

::

    示例 1:
        输入是 N-D Tensor 类型：
            input = [ [1, 2, 3, 4], [5, 6, 7, 8] ]

        输出 shape:
            input.shape = [2, 4]

    示例 2:
        输入是 SelectedRows 类型：
            input.rows = [0, 4, 19]
            input.height = 20
            input.value = [ [1, 2], [3, 4], [5, 6] ]  # inner tensor
        输出 shape:
            input.shape = [3, 2]

参数
::::::::::::

        - **input** （Tensor）-  输入的多维 Tensor 或 SelectedRows，数据类型为 bfloat16，float16，float32，float64，int32，int64。如果输入是 SelectedRows 类型，则返回其内部持有 Tensor 的 shape。


返回
::::::::::::
 Tensor，表示输入 Tensor 或 SelectedRows 的 shape。


代码示例
::::::::::::

COPY-FROM: paddle.shape
