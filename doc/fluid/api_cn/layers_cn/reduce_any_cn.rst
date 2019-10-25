.. _cn_api_fluid_layers_reduce_any:

reduce_any
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_any(input, dim=None, keep_dim=False, name=None)

该OP是对指定维度上的Tensor元素进行或逻辑（|）计算，并输出相应的计算结果。

参数：
    - **input** （Variable）— 输入变量为多维Tensor或LoDTensor，数据类型需要为bool类型。
    - **dim** （list | int，可选）— 与逻辑运算的维度。如果为None，则计算所有元素的与逻辑并返回包含单个元素的Tensoe变量，否则必须在  :math:`[−rank(input),rank(input))` 范围内。如果 :math:`dim [i] <0` ，则维度将减小为 :math:`rank+dim[i]` 。默认值为None。
    - **keep_dim** （bool）— 是否在输出Tensor中保留减小的维度。如 keep_dim 为true，否则结果张量的维度将比输入张量小，默认值为False。
    - **name** （str，可选）— 这一层的名称（可选）。如果设置为None，则将自动命名这一层。默认值为None。

返回：在指定dim上进行或逻辑计算的Tensor，数据类型为bool类型。

返回类型：Variable，数据类型为bool类型。

**代码示例**

..  code-block:: python
     
     
        import paddle.fluid as fluid
        import paddle.fluid.layers as layers
        import numpy as np

        # x是一个布尔型Tensor，元素如下:
        #    [[True, False]
        #     [False, False]]
        x = layers.assign(np.array([[1, 0], [0, 0]], dtype='int32'))
        x = layers.cast(x, 'bool')

        out = layers.reduce_any(x)  # True
        out = layers.reduce_any(x, dim=0)  # [True, False]
        out = layers.reduce_any(x, dim=-1)  # [True, False]
        # keep_dim=False, x.shape=(2,2), out.shape=(2,)

        out = layers.reduce_any(x, dim=1,
                         keep_dim=True)  # [[True], [False]]
        # keep_dim=True, x.shape=(2,2), out.shape=(2,1)






