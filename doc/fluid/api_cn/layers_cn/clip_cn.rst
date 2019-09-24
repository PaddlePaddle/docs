.. _cn_api_fluid_layers_clip:

clip
-------------------------------

.. py:function:: paddle.fluid.layers.clip(x, min, max, name=None)

该OP对输入Tensor每个元素的数值进行裁剪，使得输出Tensor元素的数值在区间[min, max]内。具体的计算公式为如下，

.. math::
        Out=MIN(MAX(x,min),max)


参数：
        - **x** （Variable）- 1维或者多维Tensor，数据类型为float32
        - **min** （float）- 最小值，输入Tensor中小于该值的元素由min代替。
        - **max** （float）- 最大值，输入Tensor中大于该值的元素由max替换。
        - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：  对元素的数值进行裁剪之后的Tesnor，与输入x具有相同的shape和数据类型

返回类型：Variable

**代码示例：**

.. code-block:: python
    
    import paddle.fluid as fluid
    input = fluid.layers.data(
        name='data', shape=[1], dtype='float32')
    reward = fluid.layers.clip(x=input, min=-1.0, max=1.0)






