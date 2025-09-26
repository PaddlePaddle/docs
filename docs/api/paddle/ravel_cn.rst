.. _cn_api_paddle_ravel:

ravel
-------------------------------

.. py:function:: paddle.ravel(input)
返回输入张量 input 的一个展平（一维）视图。

该函数会将输入 Tensor 按行优先（C-style）顺序展平成一个 1 维张量。如果输入 Tensor 已经是连续内存，则返回的结果与原 Tensor 共享数据（即为视图）；如果不连续，则会返回一份连续的拷贝。
在 动态图 模式下，返回的 Tensor 默认与输入 Tensor 共享数据。如果需要得到一份新的数据拷贝，可以使用 Tensor.clone 方法，例如：ravel_clone_x = x.ravel().clone()。

参数
:::::::::
    - **input** (Tensor) - 输入的张量，数据类型为 float16、float32、float64、int8、int32、int64、uint8 等，维度任意。

返回
:::::::::
输出 一维 Tensor，包含输入张量的所有元素，数据类型与输入相同。

代码示例
:::::::::

COPY-FROM: paddle.ravel
