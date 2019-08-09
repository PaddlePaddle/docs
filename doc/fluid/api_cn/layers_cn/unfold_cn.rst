.. _cn_api_fluid_layers_unfold:

unfold
-------------------------------

.. py:function:: paddle.fluid.layers.unfold(x, kernel_size, strides=1, paddings=0, dilation=1, name=None)

该函数会将在输入 ``x`` 上滑动的filter block转换为一列缓存数据。对于每一个卷积过滤器下的block，下面的元素都会被重新排成一列，当滑动的卷积过滤器走过整个特征图时，将会形成一系列的列。
对于每一个输入形为[N, C, H, W]的 ``x`` ，都将会按照下面公式计算出一个形为[N, Cout, Lout]的输出。


..  math::

       dkernel[0] &= dilations[0] * (kernel\_sizes[0] - 1) + 1

       dkernel[1] &= dilations[1] * (kernel\_sizes[1] - 1) + 1

       hout &= \frac{H + paddings[0] + paddings[2] - dkernel[0]}{strides[0]} + 1

       wout &= \frac{W + paddings[1] + paddings[3] - dkernel[1]}{strides[1]} + 1

       Cout &= C * kernel\_sizes[0] * kernel\_sizes[1]

       Lout &= hout * wout

参数：
    - **x**  (Variable) – 格式为[N, C, H, W]的输入张量 
    - **kernel_size**  (int|list) – 卷积核的尺寸，应该为[k_h, k_w],或者为一个整形k，处理为[k, k]
    - **strides**  (int|list) – 卷积步长，应该为[stride_h, stride_w],或者为一个整形stride，处理为[stride, stride],默认为[1, 1]
    - **paddings** (int|list) – 每个维度的扩展, 应该为[padding_top, padding_left，padding_bottom, padding_right]或者[padding_h, padding_w]或者一个整型padding。如果给了[padding_h, padding_w],则应该被扩展为[padding_h, padding_w, padding_h, padding_w]. 如果给了一个整形的padding，则会使用[padding, padding, padding, padding]，默认为[0, 0, 0, 0]
    - **dilations** (int|list) – 卷积膨胀，应当为[dilation_h, dilation_w]，或者一个整形的dilation处理为[dilation, dilation]。默认为[1, 1]。


返回： 
    滑动block的输出张量，形状如上面所描述的[N, Cout, Lout]，Cout每一个滑动block里面值的总数，Lout是滑动block的总数.

返回类型：（Variable）

**代码示例**:

.. code-block:: python
    
    import paddle.fluid as fluid
    x = fluid.layers.data(name = 'data', shape = [3, 224, 224], dtype = 'float32')
    y = fluid.layers.unfold(x, [3, 3], 1, 1, 1)






