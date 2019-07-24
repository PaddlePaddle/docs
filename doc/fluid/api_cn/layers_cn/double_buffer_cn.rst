.. _cn_api_fluid_layers_double_buffer:

double_buffer
-------------------------------

.. py:function:: paddle.fluid.layers.double_buffer(reader, place=None, name=None)


生成一个双缓冲队列reader. 数据将复制到具有双缓冲队列的位置（由place指定），如果 ``place=none`` 则将使用executor执行的位置。

参数:
  - **reader** (Variable) – 需要wrap的reader
  - **place** (Place) – 目标数据的位置. 默认是executor执行样本的位置.
  - **name** (str) – Variable 的名字. 默认为None，不关心名称时也可以设置为None


返回： 双缓冲队列的reader


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  reader = fluid.layers.open_files(filenames=['mnist.recordio'],
           shapes=[[-1, 784], [-1, 1]],
           dtypes=['float32', 'int64'])
  reader = fluid.layers.double_buffer(reader)
  img, label = fluid.layers.read_file(reader)












