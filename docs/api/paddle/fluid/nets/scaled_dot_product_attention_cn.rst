.. _cn_api_fluid_nets_scaled_dot_product_attention:

scaled_dot_product_attention
-------------------------------


.. py:function:: paddle.fluid.nets.scaled_dot_product_attention(queries, keys, values, num_heads=1, dropout_rate=0.0)




该接口实现了的基于点积（并进行了缩放）的多头注意力（Multi-Head Attention）机制。attention可以表述为将一个查询（query）和一组键值对（key-value pair）映射为一个输出；Multi-Head Attention则是使用多路进行attention，而且对attention的输入进行了线性变换。公式如下：


.. math::
    
    MultiHead(Q, K, V ) & = Concat(head_1, ..., head_h)\\
    where \  head_i & = Attention(QW_i^Q , KW_i^K , VW_i^V )\\
    Attention(Q, K, V) & = softmax(\frac{QK^\mathrm{T}}{\sqrt{d_k}})V\\

其中，:math:`Q, K, V` 分别对应 ``queries``、 ``keys`` 和 ``values``，详细内容请参阅 `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_ 

要注意该接口实现支持的是batch形式，:math:`Attention(Q, K, V)` 中使用的矩阵乘是batch形式的矩阵乘法，参考 fluid.layers. :ref:`cn_api_fluid_layers_matmul` 。

参数
::::::::::::

    - **queries** （Variable） - 形状为 :math:`[N, L_q, d_k \times h]` 的三维Tensor，其中 :math:`N` 为batch_size， :math:`L_q` 为查询序列长度，:math:`d_k \times h` 为查询的特征维度大小，:math:`h` 为head数。数据类型为float32或float64。
    - **keys** （Variable） - 形状为 :math:`[N, L_k, d_k \times h]` 的三维Tensor，其中 :math:`N` 为batch_size， :math:`L_k` 为键值序列长度，:math:`d_k \times h` 为键的特征维度大小，:math:`h` 为head数。数据类型与 ``queries`` 相同。
    - **values** （Variable） - 形状为 :math:`[N, L_k, d_v \times h]` 的三维Tensor，其中 :math:`N` 为batch_size， :math:`L_k` 为键值序列长度，:math:`d_v \times h` 为值的特征维度大小，:math:`h` 为head数。数据类型与 ``queries`` 相同。
    - **num_heads** （int） - 指明所使用的head数。head数为1时不对输入进行线性变换。默认值为1。
    - **dropout_rate** （float） - 以指定的概率对要attention到的内容进行dropout。默认值为0，即不使用dropout。

返回
::::::::::::
 形状为 :math:`[N, L_q, d_v * h]` 的三维Tensor，其中 :math:`N` 为batch_size， :math:`L_q` 为查询序列长度，:math:`d_v * h` 为值的特征维度大小。与输入具有相同的数据类型。表示Multi-Head Attention的输出。

返回类型
::::::::::::
 Variable

抛出异常
::::::::::::
    
    - :code:`ValueError`： ``queries`` 、 ``keys`` 和 ``values`` 必须都是三维。
    - :code:`ValueError`： ``queries`` 和 ``keys`` 的最后一维（特征维度）大小必须相同。
    - :code:`ValueError`： ``keys`` 和 ``values`` 的第二维（长度维度）大小必须相同。
    - :code:`ValueError`： ``keys`` 的最后一维（特征维度）大小必须是 ``num_heads`` 的整数倍。
    - :code:`ValueError`： ``values`` 的最后一维（特征维度）大小必须是 ``num_heads`` 的整数倍。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.nets.scaled_dot_product_attention