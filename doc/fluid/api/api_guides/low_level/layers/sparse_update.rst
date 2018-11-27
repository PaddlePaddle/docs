.. _api_guide_conv:

#####
稀疏更新
#####

在paddle里，我们提供了embedding接口来支持稀疏更新。他在内部表示为lookup_table operator，他的计算原理为：

.. image:: ../../../../images/lookup_table_training.png
   :scale: 50 %

==============

embedding输入参数：
---------------------

embedding需要输入(input)，形状(size)，是否需要稀疏更新(is_sparse)，是否分布式(is_distributed)，是否padding输出(padding_idx)，参数属性(param_attr)，数据类型(dtype)来决定如何计算。

- input:

  input是一个paddle的Variable, 其内容为需要查询的id向量。
- size:
  
  size为lookup table的shape，必须为两维。以NLP应用为例，第0一般为词典的大小，第一维一般为每个词对应向量的大小。
  
- is_sparse:

  反向计算的时候梯度是否为sparse tensor。如果不设置，梯度是一个LodTensor。默认为False。
   
- is_distributed:

  标志是否是用在分布式的场景下。一般大规模稀疏更新（embedding的第0维维度很大，比如几百万以上）才需要设置。具体可以参考大规模稀疏的API guide。默认为False。

- padding_idx:

  标志需要set为0的id的值。不设置时对结果没有影响。默认为None。

- param_attr:

  设置参数的属性。可以有两种设置方式：

  #. 如果设置为一个string，即为参数的名字
  #. 可以使用 :ref:`api_fluid_paramattr` 设置更多的属性。

  默认为None。不设置时，框架将设置创建具有唯一名字的默认属性参数。

- dtype:

  标志数据的具体类型，如float或者double等。默认为float32。

  
- API汇总:
 - :ref:`api_fluid_layers_embedding`

