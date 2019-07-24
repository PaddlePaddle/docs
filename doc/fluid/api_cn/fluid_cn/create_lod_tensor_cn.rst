.. _cn_api_fluid_create_lod_tensor:


create_lod_tensor
-------------------------------

.. py:function:: paddle.fluid.create_lod_tensor(data, recursive_seq_lens, place)


该函数从一个numpy数组，列表或者已经存在的lod tensor中创建一个lod tensor。

通过一下几步实现:

1. 检查length-based level of detail (LoD,长度为基准的细节层次)，或称recursive_sequence_lengths(递归序列长度)的正确性

2. 将recursive_sequence_lengths转化为offset-based LoD(偏移量为基准的LoD)

3. 把提供的numpy数组，列表或者已经存在的lod tensor复制到CPU或GPU中(依据执行场所确定)

4. 利用offset-based LoD来设置LoD

例如：
假如我们想用LoD Tensor来承载一词序列的数据，其中每个词由一个整数来表示。现在，我们意图创建一个LoD Tensor来代表两个句子，其中一个句子有两个词，另外一个句子有三个。那么数 ``data`` 可以是一个numpy数组，形状为（5,1）。同时， ``recursive_seq_lens`` 为 [[2, 3]]，表明各个句子的长度。这个长度为基准的 ``recursive_seq_lens`` 将在函数中会被转化为以偏移量为基准的 LoD [[0, 2, 5]]。

.. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
     
        t = fluid.create_lod_tensor(np.ndarray([5, 30]), [[2, 3]], fluid.CPUPlace())

参考 :ref:`api_guide_tensor` 以获取更多关于LoD的信息。

参数:
  - **data** (numpy.ndarray|list|LoDTensor) – 容纳着待复制数据的一个numpy数组、列表或LoD Tensor
  - **recursive_seq_lens** (list) – 一组列表的列表， 表明了由用户指明的length-based level of detail信息
  - **place** (Place) – CPU或GPU。 指明返回的新LoD Tensor存储地点

返回: 一个fluid LoDTensor对象，包含数据和 ``recursive_seq_lens`` 信息











