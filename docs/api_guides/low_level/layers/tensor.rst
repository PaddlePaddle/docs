..  _api_guide_tensor:

########
张量
########

Fluid 中使用两种数据结构来承载数据，分别是 `Tensor 和 LoD_Tensor <../../../user_guides/howto/basic_concept/lod_tensor.html>`_ 。 其中 LoD-Tensor 是 Fluid 的特有概念，它在 Tensor 基础上附加了序列信息。框架中可传输的数据包括：输入、输出、网络中的可学习参数，全部统一使用 LoD-Tensor 表示，Tensor 可以看作是一种特殊的 LoD-Tensor。

下面介绍这两种数据的相关操作。

Tensor
=======

1. create_tensor
---------------------
Tensor 用于在框架中承载数据，使用 :code:`create_tensor` 可以创建一个指定数据类型的 Lod-Tensor 变量，

API reference 请参考： :ref:`cn_api_fluid_layers_create_tensor`


2. create_parameter
---------------------
神经网络的训练过程是一个对参数的学习过程，Fluid 使用 :code:`create_parameter` 创建一个可学习的参数。该参数的值可以被 operator 改变。

API reference 请参考：:ref:`cn_api_fluid_layers_create_parameter`



3. create_global_var
---------------------
Fluid 使用 :code:`create_global_var` 创建一个全局 tensor，通过此 API 可以指定被创建 Tensor 变量的数据类型、形状和值。

API reference 请参考：:ref:`cn_api_fluid_layers_create_global_var`


4. cast
---------------

Fluid 使用 :code:`cast` 将数据转换为指定类型。

API reference 请参考：:ref:`cn_api_fluid_layers_cast`


5. concat
----------------

Fluid 使用 :code:`concat` 将输入数据沿指定维度连接。

API reference 请参考：:ref:`cn_api_fluid_layers_concat`


6. sums
----------------

Fluid 使用 :code:`sums` 执行对输入数据的加和。

API reference 请参考：:ref:`cn_api_fluid_layers_sums`

7. fill_constant
-----------------

Fluid 使用 :code:`fill_constant` 创建一个具有特定形状和类型的 Tensor。可以通过 :code:`value` 设置该变量的初始值。

API reference 请参考： :ref:`cn_api_fluid_layers_fill_constant`

8. assign
---------------

Fluid 使用 :code:`assign` 复制一个变量。

API reference 请参考：:ref:`cn_api_fluid_layers_assign`

9. argmin
--------------

Fluid 使用 :code:`argmin` 计算输入 Tensor 指定轴上最小元素的索引。

API reference 请参考：:ref:`cn_api_fluid_layers_assign`

10. argmax
-----------

Fluid 使用 :code:`argmax` 计算输入 Tensor 指定轴上最大元素的索引。

API reference 请参考：:ref:`cn_api_fluid_layers_argmax`

11. argsort
------------

Fluid 使用 :code:`argsort` 对输入 Tensor 在指定轴上进行排序，并返回排序后的数据变量及其对应的索引值。

API reference 请参考： :ref:`cn_api_fluid_layers_argsort`

12. ones
-------------

Fluid 使用 :code:`ones` 创建一个指定大小和数据类型的 Tensor，且初始值为 1。

API reference 请参考： :ref:`cn_api_fluid_layers_ones`

13. zeros
---------------

Fluid 使用 :code:`zeros` 创建一个指定大小和数据类型的 Tensor，且初始值为 0。

API reference 请参考： :ref:`cn_api_fluid_layers_zeros`

14. reverse
-------------------

Fluid 使用 :code:`reverse` 沿指定轴反转 Tensor。

API reference 请参考： :ref:`cn_api_fluid_layers_reverse`



LoD-Tensor
============

LoD-Tensor 非常适用于序列数据，相关知识可以参考阅读 `LoD_Tensor <../../../user_guides/howto/basic_concept/lod_tensor.html>`_ 。

1. create_lod_tensor
-----------------------

Fluid 使用 :code:`create_lod_tensor` 基于 numpy 数组、列表或现有 LoD_Tensor 创建拥有新的层级信息的 LoD_Tensor。

API reference 请参考： :ref:`cn_api_fluid_create_lod_tensor`

2. create_random_int_lodtensor
----------------------------------

Fluid 使用 :code:`create_random_int_lodtensor` 创建一个由随机整数组成的 LoD_Tensor。

API reference 请参考： :ref:`cn_api_fluid_create_random_int_lodtensor`

3. reorder_lod_tensor_by_rank
---------------------------------

Fluid 使用 :code:`reorder_lod_tensor_by_rank` 对输入 LoD_Tensor 的序列信息按指定顺序重拍。

API reference 请参考：:ref:`cn_api_fluid_layers_reorder_lod_tensor_by_rank`
