.. _cn_api_paddle_nn_layer_rnn_RNNCellBase:

RNNCellBase
-------------------------------

.. py:class:: paddle.nn.RNNCellBase(name_scope=None, dtype='float32')



**循环神经网络单元基类**

该OP（RNNCellBase）是一个抽象表示根据输入和隐藏状态来计算输出和新状态的基本类，最适合也最常用于循环神经网络。

.. py:function:: get_initial_states(batch_ref,shape=None,dtype=None,init_value=0.,batch_dim_idx=0):

根据输入的形状，数据类型和值生成初始状态。

参数：
    - **batch_ref** (Tensor) - 一个Tensor，其形状决定了生成初始状态使用的batch_size。当batch_ref形状为d时，d[batch_dim_idx]为batch_size。
    - **shape** (list|tuple，可选) - 隐藏层的形状（可以是多层嵌套的），列表或元组的第一位为batch_size，默认为-1。shape为None时，使用state_shape(property)。默认为None。
    - **dtype** (str|list|tuple，可选) - 数据类型（可以是多层嵌套的，但嵌套结构要和shape相同。或者所有Tensor的数据类型相同时可以只输入一个dtype。）。当dtype为None且state_dtype（property）不可用时，则使用paddle默认的float类型。默认为None。
    - **init_value** (float，可选) -用于初始状态的浮点数值。默认为0。
    - **batch_dim_idx** (int，可选) - 用于指定batch_size在batch_ref的索引位置的整数值。默认为0。

返回：
    - **init_state** (Tensor|tuple|list) - 根据输出的数据类型，形状和嵌套层级返回的初始状态Tensor。
