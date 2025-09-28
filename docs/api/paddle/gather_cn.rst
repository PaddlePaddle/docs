.. _cn_api_paddle_gather:

gather
-------------------------------

.. caution::

    本接口根据输入参数的不同，包含两种不同的功能。

    下面列举的两种功能参数输入方式 **互斥**，混用非公共的参数输入方法将会导致报错，请谨慎使用。

=====

.. py:function:: paddle.gather(x, index, axis=None, name=None)

根据索引 index 获取输入 ``x`` 的指定 ``axis`` 维度的条目，并将它们拼接在一起。

.. code-block:: text

        Given:

        X = [[1, 2],
             [3, 4],
             [5, 6]]

        Index = [1, 2]

        axis = 0

        Then:

        Out = [[3, 4],
               [5, 6]]

下面展示了将一个形状为[3,2]的向量通过 gather 操作，沿着 axis = 0 维度，根据索引 Index = [1, 2] 获取对应的二维张量的例子。若索引为零维，则返回结果为一维张量，例如 Index = 0 ，沿着 axis = 0 维度，则返回位于图片上方的一维张量。

.. image:: ../../images/api_legend/gather.png
   :width: 600
   :alt: 图例

参数
::::::::::::
        - **x** (Tensor) - 输入 Tensor，秩 ``rank >= 1``，支持的数据类型包括 int32、int64、float32、float64、complex64、complex128 和 uint8 (CPU)、float16（GPU） 。
        - **index** (Tensor) - 索引 Tensor，秩 ``rank = 0`` 或者 ``rank = 1``，数据类型为 int32 或 int64。
        - **axis** (Tensor) - 指定 index 获取输入的维度，``axis`` 的类型可以是 int 或者 Tensor，当 ``axis`` 为 Tensor 的时候其数据类型为 int32 或者 int64。默认值为 None，当``axis``为 None 的时候其值为 0。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，当 index 为一维 Tensor 时，返回和输入 Tensor 的形状相同的 Tensor。当 index 为零维 Tensor 时，返回 Tensor 相对于输入 Tensor 会降维， axis 指向的维度会被降维。


代码示例
::::::::::::

COPY-FROM: paddle.gather

=====

.. py:function:: paddle.gather(input, dim, index, out=None)

PyTorch 兼容的 ``gather`` 操作：根据索引 index 获取输入 ``input`` 的指定 ``dim`` 维度的条目，并将它们拼接在一起。行为与 ``cn_api_paddle_take_along_axis`` 在 ``broadcast=False`` 情况下一致。

接口对比可见 `【torch 参数更多】torch.gather`_ 。

.. _【torch 参数更多】torch.gather: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/torch/torch.gather.html

参数
::::::::::::
        - **input** (Tensor) - 输入 Tensor，支持的数据类型包括 int32、int64、float32、float64、int16、uint8、float16（GPU）以及 bfloat16（GPU） 。
        - **dim** (int) - 指定 index 获取输入的维度，``dim`` 的类型可以是 int 或者。
        - **index** (Tensor) - 索引 Tensor，``index`` 张量的各维度需要小于等于 ``input`` 张量的各维度（除 ``dim`` 维度外），且值需要在 ``input.shape[dim]`` 范围内。数据类型为 int32 或 int64。
        - **out** (Tensor，可选) - 用于引用式传入输出值，注意：动态图下 out 可以是任意 Tensor，默认值为 None。

.. caution::

        本接口没有实现 PyTorch 的 ``sparse_grad`` 参数！梯度默认是稠密的，等效于 ``sparse_grad=False``。


返回
::::::::::::
Tensor，与 input 有数据类型，与 ``index`` 有相同的形状。

代码示例
:::::::::

见 :ref:`cn_api_paddle_take_along_axis` 的代码示例。
