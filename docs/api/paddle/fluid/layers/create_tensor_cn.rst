.. _cn_api_fluid_layers_create_tensor:

create_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.create_tensor(dtype,name=None,persistable=False)




创建数据类型为dtype的Tensor。

参数
::::::::::::

    - **dtype** (str|numpy.dtype) - 创建的Tensor的数据类型，支持数据类型为bool, float16， float32， float64， int8， int16， int32， int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **persistable** (bool，可选) - 用于设置创建的Tensor的persistable属性，若不设置则默认设置为False。

返回
::::::::::::
 创建的Tensor，数据类型为dtype。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.create_tensor