.. _cn_api_paddle_nn_RNNCellBase:

RNNCellBase
-------------------------------

.. py:class:: paddle.nn.RNNCellBase(name_scope=None, dtype='float32')



**循环神经网络单元基类**

该 OP（RNNCellBase）是一个抽象表示根据输入和隐藏状态来计算输出和新状态的基本类，最适合也最常用于循环神经网络。

.. py:function:: get_initial_states(batch_ref,shape=None,dtype=None,init_value=0.,batch_dim_idx=0):

根据输入的形状，数据类型和值生成初始状态。

参数
::::::::::::

    - **batch_ref** (Tensor) - 一个 Tensor，其形状决定了生成初始状态使用的 batch_size。当 batch_ref 形状为 d 时，d[batch_dim_idx]为 batch_size。
    - **shape** (list|tuple，可选) - 隐藏层的形状（可以是多层嵌套的），列表或元组的第一位为 batch_size，默认为-1。shape 为 None 时，使用 state_shape(property)。默认为 None。
    - **dtype** (str|list|tuple，可选) - 数据类型（可以是多层嵌套的，但嵌套结构要和 shape 相同。或者所有 Tensor 的数据类型相同时可以只输入一个 dtype。）。当 dtype 为 None 且 state_dtype（property）不可用时，则使用 paddle 默认的 float 类型。默认为 None。
    - **init_value** (float，可选) -用于初始状态的浮点数值。默认为 0。
    - **batch_dim_idx** (int，可选) - 用于指定 batch_size 在 batch_ref 的索引位置的整数值。默认为 0。

返回
::::::::::::

    - **init_state** (Tensor|tuple|list) - 根据输出的数据类型，形状和嵌套层级返回的初始状态 Tensor。
