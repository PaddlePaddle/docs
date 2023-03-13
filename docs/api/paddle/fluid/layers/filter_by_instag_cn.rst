.. _cn_api_fluid_layers_filter_by_instag:

filter_by_instag
-------------------------------

.. py:function:: paddle.fluid.layers.filter_by_instag(ins, ins_tag, filter_tag, is_lod)




此函数通过 instag 来过滤 ins batch，大量属于同样的 tags 的样本，我们可以指定我们想要的一些 tags，属于这些 tags 的样本将会被保留在输出中，其余的将会移除。比如，一个 batch 有 4 个样本，每个样本都有自己的 tag 表。

Ins   |   Ins_Tag |

|:—–:|:——:|

|  0    |   0, 1 |

|  1    |   1, 3 |

|  2    |   0, 3 |

|  3    |   2, 6 |

Lod 为[1，1，1，1]，filter tags 为[1]，从上面的定义中，带有标签[1]的样本将会通过 filter，所以，样本 0 和 1 将会通过并且出现在输出中。准确来说，如果 ``is_lod`` 为 false，它是一个等于值全为 1 的 lod_tensor 的普通的 tensor，和上面的例子很相似。

参数
::::::::::::

    - **ins** (Variable) - 输入变量(LoDTensor)，通常为 2D 向量，第一个维度可以有 lod info，也可以没有。
    - **ins_tag** (Variable) - 输入变量(LoDTensor)，通常为 1 维列表，通过 lod info 来分割。
    - **filter_tag** (Variable) - 输入变量(1D Tensor/List)，通常为持有 tags 的列表。
    - **is_lod** (Bool) – 指定样本是否为 lod tensor 的布尔值。
    - **out_val_if_empty** (Int64) - 如果 batch 内样本被全部过滤，输出会被指定成这个值。

返回
::::::::::::
过滤之后的样本（LoDTensor）和 损失权重（Tensor）。

返回类型
::::::::::::
变量（Variable）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.filter_by_instag
