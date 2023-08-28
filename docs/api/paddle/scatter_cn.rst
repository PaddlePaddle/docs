.. _cn_api_paddle_cn_scatter:

scatter
-------------------------------
.. py:function:: paddle.scatter(x, index, updates, overwrite=True, name=None)


通过基于 ``updates`` 来更新选定索引 ``index`` 上的输入来获得输出。具体行为如下：

COPY-FROM: paddle.scatter:code-example1

**Notice：**
因为 ``updates`` 的应用顺序是不确定的，因此，如果索引 ``index`` 包含重复项，则输出将具有不确定性。


参数
:::::::::
    - **x** (Tensor) - ndim> = 1 的输入 N-D Tensor。数据类型可以是 float32，float64。
    - **index** （Tensor）- 一维或者零维 Tensor。数据类型可以是 int32，int64。 ``index`` 的长度不能超过 ``updates`` 的长度，并且 ``index`` 中的值不能超过输入的长度。
    - **updates** （Tensor）- 根据 ``index`` 使用 ``update`` 参数更新输入 ``x``。当 ``index`` 为一维 tensor 时，``updates`` 形状应与输入 ``x`` 相同，并且 dim>1 的 dim 值应与输入 ``x`` 相同。当 ``index`` 为零维 tensor 时，``updates`` 应该是一个 (N-1)-D 的 Tensor，并且 ``updates`` 的第 i 个维度应该与 ``x`` 的 i+1 个维度相同。
    - **overwrite** （bool，可选）- 指定索引 ``index`` 相同时，更新输出的方式。如果为 True，则使用覆盖模式更新相同索引的输出，如果为 False，则使用累加模式更新相同索引的输出。默认值为 True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，与 x 有相同形状和数据类型。


代码示例
:::::::::

COPY-FROM: paddle.scatter
