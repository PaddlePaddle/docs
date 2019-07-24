.. _cn_api_fluid_layers_multiplex:

multiplex
-------------------------------

.. py:function:: paddle.fluid.layers.multiplex(inputs, index)

引用给定的索引变量，该层从输入变量中选择行构造Multiplex变量。

假设有 :math:`m` 个输入变量，:math:`I_{i}` 代表第i个输入变量，而且 :math:`i` is in :math:`[0,m)` 。

所有输入变量都是具有相同形状的张量 :math:`[d_0,d_1, ... ,d_R]` 。

请注意，输入张量的秩应至少为2。每个输入变量将被视为形状为 :math:`[M，N]` 的二维矩阵，其中 :math:`M` 表示 :math:`d0` ，N表示 :math:`d_1 * d_2 * ... * d_R` 。

设 :math:`I_{i}[j]` 为第i个输入变量的第j行。 给定的索引变量是具有形状[M，1]的2-D张量。 设 :math:`ID[i]` 为索引变量的第i个索引值。 然后输出变量将是一个形状为 :math:`[d_0,d_1, ... ,d_R]` 的张量。

如果将输出张量视为具有形状[M，N]的2-D矩阵,并且令O[i]为矩阵的第i行，则O[i]等于 :math:`I_{ID}[i][i]`

- Ids: 索引张量
- X[0 : N - 1]: 输出的候选张量度(N >= 2).
- 对于从 0 到 batchSize-1 的每个索引i，输出是第（Ids [i]）  张量的第i行

对于第i行的输出张量：

.. math::
            \\y[i]=x_k[i]\\

其中 :math:`y` 为输出张量， :math:`x_k` 为第k个输入张量，并且 :math:`k=Ids[i]` 。

示例：

.. code-block:: text

        例1：

        假设:

        X = [[[0,0,3,4], [0,1,3,4], [0,2,4,4], [0,3,3,4]],
             [[1,0,3,4], [1,1,7,8], [1,2,4,2], [1,3,3,4]],
             [[2,0,3,4], [2,1,7,8], [2,2,4,2], [2,3,3,4]],
             [[3,0,3,4], [3,1,7,8], [3,2,4,2], [3,3,3,4]]]

        index = [3,0,1,2]

        out:[[3 0 3 4]    // X[3,0] (3 = index[i], 0 = i); i=0
             [0 1 3 4]    // X[0,1] (0 = index[i], 1 = i); i=1
             [1 2 4 2]    // X[1,2] (0 = index[i], 2 = i); i=2
             [2 3 3 4]]   // X[2,3] (0 = index[i], 3 = i); i=3

        例2:

        假设:

        X = [[[0,0,3,4], [0,1,3,4], [0,2,4,4], [0,3,3,4]],
             [[1,0,3,4], [1,1,7,8], [1,2,4,2], [1,3,3,4]]]

        index = [1,0]

        out:[[1 0 3 4]    // X[1,0] (3 = index[0], 0 = i); i=1
             [0 1 3 4]    // X[0,1] (0 = index[1], 1 = i); i=2
             [0 2 4 4]    // X[0,2] (0 = 0, 2 = i); i=3
             [0 3 3 4]]   // X[0,3] (0 = 0, 3 = i); i=4





参数:
  - **inputs** （list） - 要从中收集的变量列表。所有变量的形状相同，秩至少为2
  - **index** （Variable） -  Tensor <int32>，索引变量为二维张量，形状[M, 1]，其中M为批大小。

返回：multiplex 张量

**代码示例**

..  code-block:: python

   import paddle.fluid as fluid

   x1 = fluid.layers.data(name='x1', shape=[4], dtype='float32')
   x2 = fluid.layers.data(name='x2', shape=[4], dtype='float32')
   index = fluid.layers.data(name='index', shape=[1], dtype='int32')
   out = fluid.layers.multiplex(inputs=[x1, x2], index=index)









