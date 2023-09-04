.. _cn_api_paddle_create_parameter:

create_parameter
-------------------------------


.. py:function:: paddle.create_parameter(shape,dtype,name=None,attr=None,is_bias=False,default_initializer=None)


创建一个参数。该参数是一个可学习的变量，拥有梯度并且可优化。

.. note::
    这是一个低级别的 API。如果您希望自己创建新的 op，这个 API 将非常有用，无需使用 layers。**

参数
::::::::::::

    - **shape** (list[int]) - 指定输出 Tensor 的形状，它可以是一个整数列表。
    - **dtype** (str|numpy.dtype) – 初始化数据类型。可设置的字符串值有："float16"，"float32"，"float64"。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **attr** (ParamAttr，可选) - 指定参数的属性对象。具体用法请参见 :ref:`cn_api_fluid_ParamAttr`。默认值为 None，表示将采用 ParamAttr 的默认方式初始化。
    - **is_bias** (bool，可选) - 当 default_initializer 为空，该值会对选择哪个默认初始化程序产生影响。如果 is_bias 为真，则使用 initializer.Constant(0.0)，否则使用 Xavier()，默认值 False。
    - **default_initializer** (Initializer，可选) - 参数的初始化程序，默认值为空。

返回
::::::::::::
创建的 Parameter 变量。

代码示例
::::::::::::

COPY-FROM: paddle.static.create_parameter
