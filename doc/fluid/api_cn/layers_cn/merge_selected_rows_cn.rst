.. _cn_api_fluid_layers_merge_selected_rows:

merge_selected_rows
-------------------------------

.. py:function:: paddle.fluid.layers.merge_selected_rows(x, name=None)

**实现合并选中行（row）操作**

该运算用于合并（值相加）输入张量中重复的行。输出行没有重复的行，并且按值从小到大顺序重新对行排序。

::

    例如：

          输入:
               X.rows = [0, 5, 5, 4, 19]
               X.height = 20
               X.value = [[1, 1] [2, 2] [3, 3] [4, 4] [6, 6]]


          输出：
               Out.row is [0, 4, 5, 19]
               Out.height is 20
               Out.value is: [[1, 1] [4, 4] [5, 5] [6, 6]]



参数:
  - x (Variable) – 输入类型为SelectedRows, 选中行有可能重复
  - name (basestring|None) – 输出变量的命名

返回: 输出类型为SelectedRows，并且选中行不会重复

返回类型: 变量（Variable）

**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  b = fluid.default_main_program().global_block()
  var = b.create_var(
        name="X", dtype="float32", persistable=True,
        type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
  y = fluid.layers.merge_selected_rows(var)









