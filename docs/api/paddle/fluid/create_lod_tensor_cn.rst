.. _cn_api_fluid_create_lod_tensor:


create_lod_tensor
-------------------------------

.. py:function:: paddle.fluid.create_lod_tensor(data, recursive_seq_lens, place)




从一个numpy数组、list或LoDTensor创建一个新的LoDTensor。

具体实现方法如下：

1. 检查基于序列长度的LoD（length-based LoD），即参数中的 :code:`recursive_seq_lens` 是否正确。

2. 将 :code:`recursive_seq_lens` 转换为基于偏移量的LoD（offset-based LoD）。

3. 根据place参数，把所提供的 :code:`data` （numpy数组、list或LoDTensor）的数据复制到CPU或GPU上。

4. 将基于偏移量的LoD设置到输出的LoDTensor中。

假设我们想创建一个LoDTensor表示词的序列，其中每个词用一个整数id表示。若待创建的LoDTensor表示2个句子，其中一个句子包含2个单词，另一个句子包含3个单词。

那么，:code:`data` 为一个维度为(5, 1)的numpy整数数组；:code:`recursive_seq_lens` 为[[2, 3]]，表示每个句子含的单词个数。在该接口内部，基于序列长度的
:code:`recursive_seq_lens` [[2, 3]]会转换为为基于偏移量的LoD [[0, 2, 5]]。

请查阅 :ref:`cn_user_guide_lod_tensor` 了解更多关于LoD的介绍。

参数
::::::::::::

    - **data** (numpy.ndarray|list|LoDTensor) - 表示LoDTensor数据的numpy数组、list或LoDTensor。
    - **recursive_seq_lens** (list[list[int]]) - 基于序列长度的LoD信息。
    - **place** (CPUPlace|CUDAPlace) - 表示返回的LoDTensor存储在CPU或GPU place中。

返回
::::::::::::
 包含数据信息和序列长度信息的LoDTensor。

返回类型
::::::::::::
 LoDTensor

代码示例
::::::::::::

COPY-FROM: paddle.fluid.create_lod_tensor