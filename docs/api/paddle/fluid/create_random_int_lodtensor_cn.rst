.. _cn_api_fluid_create_random_int_lodtensor:


create_random_int_lodtensor
-------------------------------


.. py:function:: paddle.fluid.create_random_int_lodtensor(recursive_seq_lens, base_shape, place, low, high)




创建一个包含随机整数的LoDTensor。

具体实现方法如下：

1. 基于序列长度 :code:`recursive_seq_lens` 和 :code:`base_shape` 产生返回值的维度。返回值的第一维等于序列总长度，其余维度为 :code:`base_shape` 。

2. 创建一个包含随机整数的numpy数组，并作为 :code:`data` 参数传入 :ref:`cn_api_fluid_create_lod_tensor` 接口中创建LoDTensor返回。

假设我们想创建一个LoDTensor表示序列信息，共包含2个序列，维度分别为[2, 30]和[3, 30]，那么序列长度 :code:`recursive_seq_lens` 传入[[2, 3]]，:code:`base_shape` 传入[30]（即除了序列长度以外的维度）。
最后返回的LoDTensor的维度为[5, 30]，其中第一维5为序列总长度，其余维度为 :code:`base_shape` 。

参数
::::::::::::

    - **recursive_seq_lens** (list[list[int]]) - 基于序列长度的LoD信息。
    - **base_shape** (list[int]) - 除第一维以外输出结果的维度信息。
    - **place** (CPUPlace|CUDAPlace) - 表示返回的LoDTensor存储在CPU或GPU place中。
    - **low** (int) - 随机整数的下限值。
    - **high** (int) - 随机整数的上限值，必须大于或等于low。

返回
::::::::::::
 包含随机整数数据信息和序列长度信息的LoDTensor，数值范围在[low, high]之间。

返回类型
::::::::::::
 LoDTensor

代码示例
::::::::::::

COPY-FROM: paddle.fluid.create_random_int_lodtensor