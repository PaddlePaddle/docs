.. _cn_api_paddle_unique:

unique
-------------------------------

.. py:function:: paddle.unique(x, return_index=False, return_inverse=False, return_counts=False, axis=None, dtype="int64", name=None)

返回 Tensor 按升序排序后的独有元素。

参数
::::::::::::

    - **x** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64、int32、int64。
    - **return_index** (bool，可选) - 如果为 True，则还返回独有元素在输入 Tensor 中的索引。
    - **return_inverse** (bool，可选) - 如果为 True，则还返回输入 Tensor 的元素对应在独有元素中的索引，该索引可用于重构输入 Tensor。
    - **return_counts** (bool，可选) - 如果为 True，则还返回每个独有元素在输入 Tensor 中的个数。
    - **axis** (int，可选) - 指定选取独有元素的轴。默认值为 None，将输入平铺为 1-D 的 Tensor 后再选取独有元素。
    - **dtype** (np.dtype|str，可选) - 用于设置 ``index`` ， ``inverse`` 或者 ``counts`` 的类型，应该为 int32 或者 int64。默认：int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::

    - **out** (Tensor) - 独有元素构成的 Tensor，数据类型与输入一致。
    - **index** (Tensor，可选) - 独有元素在输入 Tensor 中的索引，仅在 ``return_index`` 为 True 时返回。
    - **inverse** (Tensor，可选) - 输入 Tensor 的元素对应在独有元素中的索引，仅在 ``return_inverse`` 为 True 时返回。
    - **counts** (Tensor，可选) - 每个独有元素在输入 Tensor 中的个数，仅在 ``return_counts`` 为 True 时返回。

代码示例
::::::::::::

COPY-FROM: paddle.unique


**示例图解说明**：

    图一展示了代码中一维张量通过 unique 操作去重排序后得到的新的一维张量，indices 是新的张量各个元素在原张量的索引，inverse 是原张量的各个元素在新的张量中的索引，counts 是张量中各个元素出现的次数。

    .. figure:: ../../images/api_legend/unique_1.png
       :width: 500
       :alt: 图一：一维张量示例
       :align: center

    图二展示了代码中形状为[3,3]的二维张量通过 unique 操作(axis=0)去重排序后得到新的形状为[2,3]的二维张量，新的二维张量会按照字典序进行排列。
    .. figure:: ../../images/api_legend/unique_2.png
       :width: 500
       :alt: 图二：二维张量 axis=0 示例
       :align: center

    图三展示了代码中形状为[3,3]的二维张量通过 unique 操作(axis=1)去重排序后得到新的形状为[3,3]的二维张量。由于没有重复的列向量，因此只会进行列的字典排序，注意输出的时候是按照行优先的顺序进行输出。
    .. figure:: ../../images/api_legend/unique_3.png
       :width: 500
       :alt: 图三：二维张量 axis=1 示例
       :align: center
