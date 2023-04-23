.. _cn_api_fluid_layers_merge_selected_rows:

merge_selected_rows
-------------------------------

.. py:function:: paddle.fluid.layers.merge_selected_rows(x, name=None)




累加合并 SelectedRows ( ``x`` ) 中的重复行，并对行值由小到大重新排序。

参数
::::::::::::

  - x (Variable)：类型为 SelectedRows，选中行允许重复。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

  - 含有 SelectedRows 的 Variable，选中行不重复。

返回类型
::::::::::::

  - Variable（变量）。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.merge_selected_rows