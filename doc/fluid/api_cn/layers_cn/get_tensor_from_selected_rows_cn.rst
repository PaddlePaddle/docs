.. _cn_api_fluid_layers_get_tensor_from_selected_rows:

get_tensor_from_selected_rows
-------------------------------

.. py:function::  paddle.fluid.layers.get_tensor_from_selected_rows(x, name=None)

该OP从SelectedRows类型的输入中获取向量数据，以LoDTensor的形式输出。


::

    例如：

          输入为SelectedRows类型:
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


参数：
  - **x** (SelectedRows) - SelectedRows类型的输入。
  - **name** (str) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回： 从SelectedRows中转化而来的LoDTensor，数据类型和输入一致。

返回类型： Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    b = fluid.default_main_program().global_block()
    input = b.create_var(name="X", dtype="float32", persistable=True, type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
    out = fluid.layers.get_tensor_from_selected_rows(input)









