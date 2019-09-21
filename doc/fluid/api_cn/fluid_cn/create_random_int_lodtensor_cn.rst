.. _cn_api_fluid_create_random_int_lodtensor:


create_random_int_lodtensor
-------------------------------

.. py:function:: paddle.fluid.create_random_int_lodtensor(recursive_seq_lens, base_shape, place, low, high)

创建一个包含随机整数的LoDTensor。

具体实现方法如下：

1. 基于序列长度 :code:`recursive_seq_lens` 和 :code:`base_shape` 产生返回值的维度。返回值的第一维等于序列总长度，其余维度为 :code:`base_shape` 。

2. 传建一个包含随机整数的numpy数组，并作为 :code:`data` 参数传入 :ref:`api_fluid_create_lod_tensor` 接口中创建LoDTensor返回。

假设我们想创建一个LoDTensor表示词的序列，其中每个词用一个整数id表示。若待创建的LoDTensor表示2个句子，其中一个句子包含2个单词，另一个句子包含3个单词。
那么 :code:`base_shape` 为[1]，序列长度 :code:`recursive_seq_lens` 传入[[2, 3]]。最后返回的LoDTensor的维度维[5, 1]，其中第一维5为序列总长度，其余维度为 :code:`base_shape` 。

参数:
    - **recursive_seq_lens** (list[list[int]]) - 基于序列长度的LoD信息。
    - **base_shape** (list) - 除第一维以外输出结果的维度信息。
    - **place** (CPUPlace|CUDAPlace) - 表示返回的LoDTensor存储在CPU或GPU place中。
    - **low** (int) - 随机整数的下限值。
    - **high** (int) - 随机整数的上限值。

返回: 包含随机整数数据信息和序列长度信息的LoDTensor。

返回类型: LoDTensor

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
     
        t = fluid.create_random_int_lodtensor(recursive_seq_lens=[[2, 3]],base_shape=[30], place=fluid.CPUPlace(), low=0, high=10)

