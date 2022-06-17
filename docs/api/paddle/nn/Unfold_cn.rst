.. _cn_api_fluid_layers_unfold:

unfold
-------------------------------

.. py:function:: paddle.nn.Unfold(kernel_size=[3,3], strides=1, paddings=1, dilation=1, name=None)




该OP实现的功能与卷积中用到的im2col函数一样，通常也被称作为im2col过程。对于每一个卷积核覆盖下的区域，元素会被重新排成一列。当卷积核在整个图片上滑动时，将会形成一系列的列向量。对于每一个输入形状为[N, C, H, W]的 ``x``，都将会按照下面公式计算出一个形状为[N, Cout, Lout]的输出。

.. note::
   对应的 `functional方法` 请参考：:ref:`cn_api_nn_functional_unfold` 。


**样例**：

::

      Given:
        x.shape = [5, 10, 25, 25]
        kernel_size = [3, 3]
        strides = 1
        paddings = 1

      Return:
        out.shape = [5, 90, 625]


参数
::::::::::::

    - **kernel_size**  (int|list of int) – 卷积核的尺寸，整数或者整型列表。如果为整型列表，应包含两个元素 ``[k_h, k_w]``，卷积核大小为 ``k_h * k_w``；如果为整数k，会被当作整型列表 ``[k, k]`` 处理
    - **strides**  (int|list of int，可选) – 卷积步长，整数或者整型列表。如果为整型列表，应该包含两个元素 ``[stride_h, stride_w]``。如果为整数，则 ``stride_h = stride_w = strides``。默认值为1
    - **paddings** (int|list of int，可选) – 每个维度的扩展，整数或者整型列表。如果为整型列表，长度应该为4或者2；长度为4 对应的padding参数是：[padding_top, padding_left，padding_bottom, padding_right]，长度为2对应的padding参数是[padding_h, padding_w]，会被当作[padding_h, padding_w, padding_h, padding_w]处理。如果为整数padding，则会被当作[padding, padding, padding, padding]处理。默认值为0
    - **dilations** (int|list of int，可选) – 卷积膨胀，整型列表或者整数。如果为整型列表，应该包含两个元素[dilation_h, dilation_w]。如果是整数dilation，会被当作整型列表[dilation, dilation]处理。默认值为1
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
 - **输入** : 4-D Tensor，形状为[N, C, H, W]，数据类型为float32或者float64
 - **输出**：形状如上面所描述的[N, Cout, Lout]，Cout每一个滑动block里面覆盖的元素个数，Lout是滑动block的个数，数据类型与 ``x`` 相同


代码示例
::::::::::::

.. code-block:: python
    
   import paddle
   import paddle.nn as nn

   x = paddle.randn((100,3,224,224))
   unfold = nn.Unfold(kernel_sizes=[3, 3])
   result = unfold(x)
   print(result)






