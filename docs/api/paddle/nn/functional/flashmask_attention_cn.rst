.. _cn_api_paddle_nn_functional_flashmask_attention:

flashmask_attention
-------------------------------

.. py:function:: paddle.nn.functional.flashmask_attention(query, key, value, startend_row_indices, dropout=0.0, causal=False, return_softmax_lse=False, return_seed_offset=False, fixed_seed_offset=None, rng_name="", training=True, name=None)

.. math::

    result = softmax(\frac{ Q * K^T }{\sqrt{d}} + mask) * V

用稀疏的 flashmask 表达的 flash_attention。
flashmask 将通过参数 :code:`startend_row_indices` 表示作用在 Attention Score 矩阵上的 mask ， Attention Score 矩阵指的是 :math:`Q * K^T` ，元素被 mask 指的是将 Score 矩阵中对应位置设置为 :math:`-inf` 。

下图展示了多种 mask 的示例，图中为 Score 矩阵，灰色区域元素表示被 mask ，上方数字表示 startend_row_indices 的值，一行数字表明 startend_row_indices 的 shape 为 [batch_size, num_heads, seq_len, 1] ，二行数字表明 startend_row_indices 的 shape 为 [batch_size, num_heads, seq_len, 2] ， 四行数字表明 startend_row_indices 的 shape 为 [batch_size, num_heads, seq_len, 4] 。

.. image:: ../../../../images/FlashMask1.png
   :width: 900
   :alt: pipeline
   :align: center

图(a)中 :code:`causal=True` ， :code:`startend_row_indices` 的值如下

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
          [[[[5 ],
              [5 ],
              [5 ],
              [5 ],
              [5 ],
              [5 ],
              [5 ],
              [5],
              [5],
              [5]]]])

图(b)中 :code:`causal=True` ， :code:`startend_row_indices` 的值如下

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
          [[[[4 ],
              [4 ],
              [4 ],
              [4 ],
              [7 ],
              [7 ],
              [7 ],
              [10],
              [10],
              [10]]]])

.. image:: ../../../../images/FlashMask2.png
   :width: 900
   :alt: pipeline
   :align: center

图(c)中 :code:`causal=True` ， :code:`startend_row_indices` 的值如下

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
          [[[[10 ],
              [10 ],
              [10 ],
              [10 ],
              [7 ],
              [7 ],
              [7 ],
              [10],
              [10],
              [10]]]])

图(d)中 :code:`causal=True` ， :code:`startend_row_indices` 的值如下

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
          [[[[10 ],
              [4 ],
              [5 ],
              [6 ],
              [7 ],
              [8 ],
              [9 ],
              [10],
              [10],
              [10]]]])

.. image:: ../../../../images/FlashMask3.png
   :width: 900
   :alt: pipeline
   :align: center

图(e)中 :code:`causal=True` ， :code:`startend_row_indices` 的值如下

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
          [[[[4 , 7 ],
              [4 , 7 ],
              [4 , 7 ],
              [4 , 7 ],
              [10, 10],
              [10, 10],
              [10, 10],
              [10, 10],
              [10, 10],
              [10, 10]]]])

图(f)中 :code:`causal=False` ， :code:`startend_row_indices` 的值如下

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
          [[[[4 , 0 ],
              [4 , 0 ],
              [4 , 0 ],
              [4 , 0 ],
              [7, 4],
              [7, 4],
              [7, 4],
              [10, 7],
              [10, 7],
              [10, 7]]]])

.. image:: ../../../../images/FlashMask4.png
   :width: 900
   :alt: pipeline
   :align: center

图(g)中 :code:`causal=False` ， :code:`startend_row_indices` 的值如下

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 4], dtype=int32, place=Place(gpu:0), stop_gradient=True,
          [[[[10, 10, 0 , 0 ],
              [10, 10, 0 , 0 ],
              [10, 10, 0 , 0 ],
              [3 , 10, 0 , 0 ],
              [4 , 10, 3 , 4 ],
              [5 , 10, 3 , 5 ],
              [6 , 10, 3 , 6 ],
              [7 , 10, 3 , 7 ],
              [8 , 10, 3 , 8 ],
              [9 , 10, 3 , 9 ]]]])

图(h)中 :code:`causal=True` ， :code:`startend_row_indices` 的值如下

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
          [[[[10 ],
              [4 ],
              [8 ],
              [6 ],
              [10 ],
              [7 ],
              [10 ],
              [9],
              [10],
              [10]]]])

参数
::::::::::::

    - **query** (Tensor) - 输入 Query Tensor，shape =[batch_size, seq_len, num_heads, head_dim]，数据类型为 float16 或 bfloat16。
    - **key** (Tensor) - 输入 Key Tensor，shape 以及 dtype 和 query 相同。
    - **value** (Tensor) - 输入 Value Tensor，shape 以及 dtype 和 query 相同。
    - **startend_row_indices** (Tensor)
            - 稀疏掩码索引，shape 为 [batch_size, num_heads, seq_len, {1, 2, 4}]，数据类型为 int32。
                                       num_heads 为 1 或与 k 的 num_heads 相同，num_heads 取 1 时将被广播到与 k 的 num_heads 相同。
                                       根据 causal 参数的取值不同，startend_row_indices 可取不同形状并具有不同含义, startend_row_indices 中的值将依次被记为 r1,r2,r3,r4。
            - 当 :code:`causal=True` 且 shape 取 [batch_size, num_heads, seq_len, 1] 时,
              startend_row_indices 的值 r1 表示 Score 矩阵中左下三角从第 r1 行下方（包括）的元素将被 mask
            - 当 :code:`causal=True` 且 shape 取 [batch_size, num_heads, seq_len, 2] 时,
              startend_row_indices 的值 r1,r2 表示 Score 矩阵中左下三角从第 r1 行下方（包括）但在第 r2 行上方（不包括）的元素将被 mask
            - 当 :code:`causal=False` 且 shape 取 [batch_size, num_heads, seq_len, 2] 时,
              startend_row_indices 的值 r1,r2 表示 Score 矩阵中左下三角从第 r1 行下方（包括）的元素将被 mask，右上三角从第 r2 行上方（不包括）的元素将被 mask
            - 当 :code:`causal=False` 且 shape 取 [batch_size, num_heads, seq_len, 4] 时 （尚未支持）,
              startend_row_indices 的值 r1,r2,r3,r4 表示 Score 矩阵中左下三角从第 r1 行下方（包括）但在第 r2 行上方（不包括）的元素将被 mask，右上三角从第 r3 行下方（包括）但在第 r4 行上方（不包括）的元素将被 mask。
    - **dropout** (bool，可选) – dropout 概率值，默认值为 0。
    - **causal** (bool，可选) - 是否使用 causal 模式，默认值：False。
    - **window_size** (int|tuple, 可选) - 表示滑动窗口局部注意力的窗口大小。
                如果causal为True，位置 i 处的 Query 只与在 [i - window_size, i] 或 [i - window_size[0], i] 范围内的Key形成Attention。
                如果causal为False，位置 i 处的 Query 只与在 [i - window_size, i + window_size] 或 [i - window_size[0], i + window_size[1]] 范围内的Key形成Attention。
    - **return_softmax_lse** (bool，可选) - 是否返回 softmax_lse 的结果。默认值为 False，表示不返回 :code:`softmax_lse` 。
    - **return_seed_offset** (bool，可选) - 是否返回 seed_offset 的结果。默认值为 False，表示不返回 :code:`seed_offset` 。
    - **fixed_seed_offset** (Tensor，可选) - 固定 Dropout 的 offset seed。 默认值为 None, 表示不固定 seed。
    - **rng_name** (str，可选) - 随机数生成器名称。 默认值为 ""。
    - **training** (bool，可选) - 指示是否为训练模式。默认值为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
`Tensor`，attention 的结果。

`softmax_lse`，当 return_softmax_lse 为 True 时，返回的 softmax_lse 的值

`seed_offset`，当 return_seed_offset 为 True 时，返回的 seed_offset 的值


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.flashmask_attention
