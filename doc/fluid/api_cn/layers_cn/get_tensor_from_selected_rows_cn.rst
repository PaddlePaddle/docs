.. _cn_api_fluid_layers_get_tensor_from_selected_rows:

get_tensor_from_selected_rows
-------------------------------

.. py:function::  paddle.fluid.layers.get_tensor_from_selected_rows(x, name=None)

:code:`Get Tensor From Selected Rows` 用于从选中行（Selected Rows）中获取张量

参数：
  - **x** (Variable) - 输入，类型是SelectedRows
  - **name** (basestring|None) - 输出的名称

返回： 输出类型为LoDTensor

返回类型： out(Variable)

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    b = fluid.default_main_program().global_block()
    input = b.create_var(name="X", dtype="float32", persistable=True, type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
    out = fluid.layers.get_tensor_from_selected_rows(input)









