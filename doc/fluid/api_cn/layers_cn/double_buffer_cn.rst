.. _cn_api_fluid_layers_double_buffer:

double_buffer
-------------------------------

.. py:function:: paddle.fluid.layers.double_buffer(reader, place=None, name=None)


生成一个双缓冲队列reader。数据将复制到具有双缓冲队列的位置（由place指定），如果 ``place=None`` 则将使用executor执行的位置。

参数:
    - **reader** (Variable) – 需要wrap的reader变量Reader。
    - **place** (Place，可选) – 目标数据的位置，比如CPU，GPU，GPU需要指明是哪张卡。默认是executor执行样本的位置。
    - **name** (str，可选) – 变量的名字。该参数供开发人员打印调试信息时使用，具体用法参见 :ref:`api_guide_Name`，默认值为None。



返回：Variable(Reader)。双缓冲队列的reader。

返回类型：变量(Variable)。


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  reader = fluid.layers.open_files(filenames=['mnist.recordio'],
           shapes=[[-1, 784], [-1, 1]],
           lod_levels=[0, 0],
           dtypes=['float32', 'int64'])
  reader = fluid.layers.double_buffer(reader)
  img, label = fluid.layers.read_file(reader)












