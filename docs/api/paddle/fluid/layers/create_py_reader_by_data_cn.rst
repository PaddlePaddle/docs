.. _cn_api_fluid_layers_create_py_reader_by_data:

create_py_reader_by_data
-------------------------------


.. py:function:: paddle.fluid.layers.create_py_reader_by_data(capacity,feed_list,name=None,use_double_buffer=True)




创建一个 Python 端提供数据的 reader。该 OP 与 :ref:`cn_api_fluid_layers_py_reader` 类似，不同点在于它能够从 feed 变量列表读取数据。

参数
::::::::::::

  - **capacity** (int) - ``py_reader`` 维护的队列缓冲区的容量大小。单位是 batch 数量。若 reader 读取速度较快，建议设置较大的 ``capacity`` 值。
  - **feed_list** (list(Variable)) - feed 变量列表，这些变量一般由 :code:`fluid.data()` 创建。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **use_double_buffer** (bool，可选) - 是否使用双缓冲区，双缓冲区是为了预读下一个 batch 的数据、异步 CPU -> GPU 拷贝。默认值为 True。

返回
::::::::::::
能够从 feed 变量列表读取数据的 reader，数据类型和 feed 变量列表中变量的数据类型相同。

返回类型
::::::::::::
reader

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.create_py_reader_by_data
