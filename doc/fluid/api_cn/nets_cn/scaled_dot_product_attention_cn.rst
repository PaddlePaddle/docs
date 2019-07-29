.. _cn_api_fluid_nets_scaled_dot_product_attention:

scaled_dot_product_attention
-------------------------------

.. py:function:: paddle.fluid.nets.scaled_dot_product_attention(queries, keys, values, num_heads=1, dropout_rate=0.0)

点乘attention运算。

attention运算机制可以被视为将查询和一组键值对映射到输出。 将输出计算为值的加权和，其中分配给每个值的权重由查询的兼容性函数（此处的点积）与对应的密钥计算。

可以通过（batch）矩阵乘法实现点积attention运算，如下所示：

.. math::
      Attention(Q, K, V)= softmax(QK^\mathrm{T})V

请参阅 `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_ 

参数：
         - **queries** （Variable） - 输入变量，应为3-D Tensor。
         - **keys** （Variable） - 输入变量，应为3-D Tensor。
         - **values** （Variable） - 输入变量，应为3-D Tensor。
         - **num_heads** （int） - 计算缩放点积attention运算的head数。默认值：1。
         - **dropout_rate** （float） - 降低attention的dropout率。默认值：0.0。

返回：   通过multi-head来缩放点积attention运算的三维张量。

返回类型：  变量（Variable）。

抛出异常:    
    - ``ValueError`` - 如果输入查询键，值不是3-D Tensor会报错。

.. note::
    当num_heads> 1时，分别学习三个线性投影，以将输入查询，键和值映射到查询'，键'和值'。 查询'，键'和值'与查询，键和值具有相同的形状。
    当num_heads == 1时，scaled_dot_product_attention没有可学习的参数。

**代码示例**

.. code-block:: python

          import paddle.fluid as fluid
          
          queries = fluid.layers.data(name="queries",
                                      shape=[3, 5, 9],
                                      dtype="float32",
                                      append_batch_size=False)
          queries.stop_gradient = False
          keys = fluid.layers.data(name="keys",
                                   shape=[3, 6, 9],
                                   dtype="float32",
                                   append_batch_size=False)
          keys.stop_gradient = False
          values = fluid.layers.data(name="values",
                                     shape=[3, 6, 10],
                                     dtype="float32",
                                     append_batch_size=False)
          values.stop_gradient = False
          contexts = fluid.nets.scaled_dot_product_attention(queries, keys, values)
          contexts.shape  # [3, 5, 10]









