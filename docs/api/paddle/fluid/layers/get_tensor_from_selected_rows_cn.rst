.. _cn_api_fluid_layers_get_tensor_from_selected_rows:

get_tensor_from_selected_rows
-------------------------------

.. py:function::  paddle.fluid.layers.get_tensor_from_selected_rows(x, name=None)




该OP从SelectedRows类型的输入中获取向量数据，以LoDTensor的形式输出。


::

    例如：

          输入为SelectedRows类型：
               x.rows = [0, 5, 5, 4, 19]
               x.height = 20
               x.value = [[1, 1] [2, 2] [2, 2] [3, 3] [6, 6]]

          输出为LoDTensor：
               out.shape = [5, 2]
               out.data = [[1, 1],
                           [2, 2],
                           [2, 2],
                           [3, 3],
                           [6, 6]]


参数
::::::::::::

  - **x** (SelectedRows) - SelectedRows类型的输入，数据类型为float32，float64，int32或int64。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 从SelectedRows中转化而来的LoDTensor，数据类型和输入一致。

返回类型
::::::::::::
 Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.get_tensor_from_selected_rows