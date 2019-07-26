.. _cn_api_fluid_create_random_int_lodtensor:


create_random_int_lodtensor
-------------------------------

.. py:function:: paddle.fluid.create_random_int_lodtensor(recursive_seq_lens, base_shape, place, low, high)



该函数创建一个存储多个随机整数的LoD Tensor。

该函数是经常在书中出现的案例，所以我们根据新的API： ``create_lod_tensor`` 更改它然后放在LoD Tensor板块里来简化代码。

该函数实现以下功能：

1. 根据用户输入的length-based ``recursive_seq_lens`` （基于长度的递归序列长）和在 ``basic_shape`` 中的基本元素形状计算LoDTensor的整体形状
2. 由此形状，建立numpy数组
3. 使用API： ``create_lod_tensor`` 建立LoDTensor


假如我们想用LoD Tensor来承载一词序列，其中每个词由一个整数来表示。现在，我们意图创建一个LoD Tensor来代表两个句子，其中一个句子有两个词，另外一个句子有三个。那么 ``base_shape`` 为[1], 输入的length-based ``recursive_seq_lens`` 是 [[2, 3]]。那么LoDTensor的整体形状应为[5, 1]，并且为两个句子存储5个词。

参数:
    - **recursive_seq_lens** (list) – 一组列表的列表， 表明了由用户指明的length-based level of detail信息
    - **base_shape** (list) – LoDTensor所容纳的基本元素的形状
    - **place** (Place) –  CPU或GPU。 指明返回的新LoD Tensor存储地点
    - **low** (int) – 随机数下限
    - **high** (int) – 随机数上限

返回: 一个fluid LoDTensor对象，包含张量数据和 ``recursive_seq_lens`` 信息

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
     
        t = fluid.create_random_int_lodtensor(recursive_seq_lens=[[2, 3]],base_shape=[30], place=fluid.CPUPlace(), low=0, high=10)

