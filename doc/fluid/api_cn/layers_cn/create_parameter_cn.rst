.. _cn_api_fluid_layers_create_parameter:

create_parameter
-------------------------------

.. py:function:: paddle.fluid.layers.create_parameter(shape,dtype,name=None,attr=None,is_bias=False,default_initializer=None)

创建一个参数。该参数是一个可学习的变量，拥有梯度并且可优化。

注：这是一个低级别的API。如果您希望自己创建新的op，这个API将非常有用，无需使用layers。

参数：
    - **shape** (list[int])-参数的维度
    - **dtype** (string)-参数的元素类型
    - **attr** (ParamAttr)-参数的属性
    - **is_bias** (bool)-当default_initializer为空，该值会对选择哪个默认初始化程序产生影响。如果is_bias为真，则使用initializer.Constant(0.0)，否则使用Xavier()。
    - **default_initializer** (Initializer)-参数的初始化程序

返回：创建的参数

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    W = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')









