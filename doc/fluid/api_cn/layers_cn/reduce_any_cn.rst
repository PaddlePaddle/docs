.. _cn_api_fluid_layers_reduce_any:

reduce_any
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_any(input, dim=None, keep_dim=False, name=None)

计算给定维度上张量（Tensor）元素的或逻辑。     

参数：
          - **input** （Variable）：输入变量为Tensor或LoDTensor。
          - **dim** （list | int | None）：或逻辑运算的维度。如果为None，则计算所有元素的或逻辑并返回仅包含单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将减小为 :math:`rank+dim[i]` 。
          - **keep_dim** （bool | False）：是否在输出Tensor中保留减小的维度。除非 ``keep_dim`` 为true，否则结果张量的维度将比输入张量小。
          - **name** （str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  减少维度之后的Tensor变量。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python
     
     
        # x是一个布尔型Tensor，元素如下:
        #    [[True, False]
        #     [False, False]]
        # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
        import paddle.fluid as fluid
        import paddle.fluid.layers as layers
        import numpy as np
        fluid.layers.reduce_any(x)  # True
        fluid.layers.reduce_any(x, dim=0)  # [True, False]
        fluid.layers.reduce_any(x, dim=-1)  # [True, False]
        fluid.layers.reduce_any(x, dim=1,
                                 keep_dim=True)  # [[True], [False]]






