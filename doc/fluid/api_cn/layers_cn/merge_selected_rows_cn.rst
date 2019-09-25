.. _cn_api_fluid_layers_merge_selected_rows:

merge_selected_rows
-------------------------------

.. py:function:: paddle.fluid.layers.merge_selected_rows(x, name=None)

累加合并 `SelectedRows <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/selected_rows.h>`_ ( ``x`` ) 中的重复行，并对行值由小到大重新排序。

参数:
  - x (Variable) : 类型为 SelectedRows，选中行允许重复。
  - name (basestring|None) : 输出变量名称。

返回:
  - 含有 SelectedRows 的 Variable，选中行不重复。

返回类型:
  - Variable（变量）。

**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  import numpy

  place = fluid.CPUPlace()
  block = fluid.default_main_program().global_block()

  var = block.create_var(name="X2",
                         dtype="float32",
                         persistable=True,
                         type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
  y = fluid.layers.merge_selected_rows(var)
  z = fluid.layers.get_tensor_from_selected_rows(y)

  x_rows = [0, 2, 2, 4, 19]
  row_numel = 2
  np_array = numpy.ones((len(x_rows), row_numel)).astype("float32")

  x = fluid.global_scope().var("X2").get_selected_rows()
  x.set_rows(x_rows)
  x.set_height(20)
  x_tensor = x.get_tensor()
  x_tensor.set(np_array, place)

  exe = fluid.Executor(place=place)
  result = exe.run(fluid.default_main_program(), fetch_list=[z])

  print("x_rows: ", x_rows)
  print("np_array: ", np_array)
  print("result: ", result)
  '''
  Output Values:
  ('x_rows: ', [0, 2, 2, 4, 19])
  ('np_array: ', array([[1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.]], dtype=float32))
  ('result: ', [array([[1., 1.],
         [2., 2.],
         [1., 1.],
         [1., 1.]], dtype=float32)])
  '''
