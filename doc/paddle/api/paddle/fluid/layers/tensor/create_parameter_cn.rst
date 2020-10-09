.. _cn_api_fluid_layers_create_parameter:

create_parameter
-------------------------------


.. py:function:: paddle.fluid.layers.create_parameter(shape,dtype,name=None,attr=None,is_bias=False,default_initializer=None)




该OP创建一个参数。该参数是一个可学习的变量, 拥有梯度并且可优化。

**注意：这是一个低级别的API。如果您希望自己创建新的op，这个API将非常有用，无需使用layers。**

参数：
    - **shape** (list[int]) - 指定输出Tensor的形状，它可以是一个整数列表。
    - **dtype** (str|numpy.dtype) – 初始化数据类型。可设置的字符串值有："float16"，"float32"，"float64"。
    - **name** (str，可选) - 参数的名称。具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    - **attr** (ParamAttr，可选) - 指定参数的属性对象。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。默认值为None，表示将采用 ParamAttr 的默认方式初始化。
    - **is_bias** (bool，可选) - 当default_initializer为空，该值会对选择哪个默认初始化程序产生影响。如果is_bias为真，则使用initializer.Constant(0.0)，否则使用Xavier()，默认值False。
    - **default_initializer** (Initializer，可选) - 参数的初始化程序，默认值为空。

返回：创建的Tensor变量。

返回类型：Variable。

**代码示例**：

.. code-block:: python

    import paddle
    W = paddle.static.create_parameter(shape=[784, 200], dtype='float32')









