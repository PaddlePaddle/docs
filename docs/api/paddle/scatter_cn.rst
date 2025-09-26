.. _cn_api_paddle_scatter:

scatter
-------------------------------

.. caution::

    本接口根据输入参数的不同，包含两种不同的功能。

    下面列举的两种功能参数输入方式 **互斥**，混用非公共的参数输入方法将会导致报错，请谨慎使用。


=====

.. py:function:: paddle.scatter(x, index, updates, overwrite=True, name=None)


通过基于  ``updates``  来更新选定索引  ``index``  上的输入来获得输出。具体行为如下：

如下图，当 overwrite 为 True 的时候使用覆盖模式更新相同索引的输出，依次将  ``x[index[i]]``  更新为  ``update[i]``  ；而当 overwrite 为 False 时使用累加模式更新相同索引的输出，先依次将  ``x[index[i]]``  更新为与该行大小相同的元素值均为 0 的 Tensor ，再依次将  ``update[i]``  加到  ``x[index[i]]``  产生输出。

.. image:: ../../images/api_legend/scatter.png
   :alt: 图例- scatter 的行为展示

COPY-FROM: paddle.scatter:scatter-example-1

**Notice：**
因为  ``updates``  的应用顺序是不确定的，因此，如果索引  ``index``  包含重复项，则输出将具有不确定性。


参数
:::::::::
    - **x** （Tensor） -  ``ndim>= 1``  的输入 N-D Tensor。数据类型可以是 float32，float64。
    - **index** （Tensor）- 一维或者零维 Tensor。数据类型可以是 int32，int64。  ``index``  的长度不能超过  ``updates``  的长度，并且  ``index``  中的值不能超过输入的长度。
    - **updates** （Tensor）- 根据  ``index``  使用  ``update``  参数更新输入  ``x`` 。当  ``index``  为一维 tensor 时， ``updates``  形状应与输入  ``x``  相同，并且  ``dim>1``  的 dim 值应与输入  ``x``  相同。当  ``index``  为零维 tensor 时， ``updates``  应该是一个  ``(N-1)-D``  的 Tensor，并且  ``updates``  的第 i 个维度应该与  ``x``  的  ``i+1``  个维度相同。
    - **overwrite** （bool，可选）- 指定索引  ``index``  相同时，更新输出的方式。如果为 True，则使用覆盖模式更新相同索引的输出，如果为 False，则使用累加模式更新相同索引的输出。默认值为 True。
    - **name** （str，可选） - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，与 x 有相同形状和数据类型。


代码示例
:::::::::

COPY-FROM: paddle.scatter:scatter-example-2


=====

.. py:function:: paddle.scatter(input, dim, index, src, reduce=None, out=None)

PyTorch 兼容的 scatter 函数。基于 :ref:`cn_api_paddle_put_along_axis` 实现，等效于  ``paddle.put_along_axis(..., broadcast=False)`` 。详细的用法见 :ref:`cn_api_paddle_put_along_axis`。

参数
:::::::::
    - **input** （Tensor） - 输入 N-D Tensor。数据类型可以是 float32，float64，float16，bfloat16，int32，int64，int16，uint8。
    - **dim** （int） - 进行 scatter 操作的维度，范围为  ``[-input.ndim, input.ndim)`` 。
    - **index** （Tensor）- 索引矩阵，包含沿轴提取 1d 切片的下标，必须和 arr 矩阵有相同的维度。注意，除了  ``dim``  维度外，  ``index``  张量的各维度大小应该小于等于  ``input``  以及  ``src``  张量。内部的值应该在  ``input.shape[dim]``  范围内。数据类型可以是 int32，int64。
    - **src** （Tensor）- 需要插入的值。 ``src``  是张量时，各维度大小需要至少大于等于  ``index``  各维度。不受到  ``input``  的各维度约束。当为标量值时，会自动广播大小到  ``index`` 。数据类型为：bfloat16、float16、float32、float64、int32、int64、uint8、int16。本参数有一个互斥的别名  ``value`` 。
    - **reduce** （str，可选）- 指定 scatter 的归约方式。默认值为 None，等效为  ``assign`` 。可选为  ``add`` 、  ``multiple`` 、  ``mean`` 、  ``amin`` 、  ``amax`` 。不同的规约操作插入值 src 对于输入矩阵 arr 会有不同的行为，如为  ``assgin``  则覆盖输入矩阵，  ``add``  则累加至输入矩阵，  ``mean``  则计算累计平均值至输入矩阵，  ``multiple``  则累乘至输入矩阵，  ``amin``  则计算累计最小值至输入矩阵，  ``amax``  则计算累计最大值至输入矩阵。
    - **out** (Tensor，可选) - 用于引用式传入输出值，注意：动态图下 out 可以是任意 Tensor，默认值为 None。

返回
:::::::::
Tensor，与 input 有相同形状和数据类型。

代码示例
:::::::::

见 :ref:`cn_api_paddle_put_along_axis` 的代码示例。
