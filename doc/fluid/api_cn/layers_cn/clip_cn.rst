.. _cn_api_fluid_layers_clip:

clip
-------------------------------

.. py:function:: paddle.fluid.layers.clip(x, min, max, name=None)

clip算子

clip算子限制给定输入的值在一个区间内。间隔使用参数"min"和"max"来指定：公式为

.. math::
        Out=min(max(X,min),max)

参数：
        - **x** （Variable）- （Tensor）clip运算的输入，维数必须在[1,9]之间。
        - **min** （FLOAT）- （float）最小值，小于该值的元素由min代替。
        - **max** （FLOAT）- （float）最大值，大于该值的元素由max替换。
        - **name** （basestring | None）- 输出的名称。

返回：        （Tensor）clip操作后的输出和输入（X）具有形状（shape）

返回类型：        输出（Variable）。

**代码示例：**

.. code-block:: python
    
    import paddle.fluid as fluid
    input = fluid.layers.data(
        name='data', shape=[1], dtype='float32')
    reward = fluid.layers.clip(x=input, min=-1.0, max=1.0)






