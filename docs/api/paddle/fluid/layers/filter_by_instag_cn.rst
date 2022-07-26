.. _cn_api_fluid_layers_filter_by_instag:

filter_by_instag
-------------------------------

.. py:function:: paddle.fluid.layers.filter_by_instag(ins, ins_tag, filter_tag, is_lod)




此函数通过instag来过滤ins batch，大量属于同样的tags的样本，我们可以指定我们想要的一些tags，属于这些tags的样本将会被保留在输出中，其余的将会移除。比如，一个batch有4个样本，每个样本都有自己的tag表。

Ins   |   Ins_Tag |

|:—–:|:——:|

|  0    |   0, 1 |

|  1    |   1, 3 |

|  2    |   0, 3 |

|  3    |   2, 6 |

Lod为[1，1，1，1]，filter tags为[1]，从上面的定义中，带有标签[1]的样本将会通过filter，所以，样本0和1将会通过并且出现在输出中。准确来说，如果 ``is_lod`` 为false，它是一个等于值全为1的lod_tensor的普通的tensor，和上面的例子很相似。

参数
::::::::::::

    - **ins** (Variable) - 输入变量(LoDTensor)，通常为2D向量，第一个维度可以有lod info，也可以没有。
    - **ins_tag** (Variable) - 输入变量(LoDTensor)，通常为1维列表，通过lod info来分割。
    - **filter_tag** (Variable) - 输入变量(1D Tensor/List)，通常为持有tags的列表。
    - **is_lod** (Bool) – 指定样本是否为lod tensor的布尔值。
    - **out_val_if_empty** (Int64) - 如果batch内样本被全部过滤，输出会被指定成这个值。
    
返回
::::::::::::
过滤之后的样本（LoDTensor）和 损失权重（Tensor）。

返回类型
::::::::::::
变量（Variable）

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.filter_by_instag