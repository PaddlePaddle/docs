.. _cn_api_fluid_layers_split:

split
-------------------------------

.. py:function:: paddle.fluid.layers.split(input,num_or_sections,dim=-1,name=None)

将输入张量分解成多个子张量

参数：
    - **input** (Variable)-输入变量，类型为Tensor或者LoDTensor
    - **num_or_sections** (int|list)-如果num_or_sections是整数，则表示张量平均划分为的相同大小子张量的数量。如果num_or_sections是一列整数，列表的长度代表子张量的数量，整数依次代表子张量的dim维度的大小
    - **dim** (int)-将要划分的维。如果dim<0,划分的维为rank(input)+dim
    - **name** (str|None)-该层名称（可选）。如果设置为空，则自动为该层命名

返回：一列分割张量

返回类型：列表(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    # 输入是维为[-1, 3,9,5]的张量：
    input = fluid.layers.data(
         name="input", shape=[3, 9, 5], dtype="float32")

    x0, x1, x2 = fluid.layers.split(x, num_or_sections=3, dim=2)
    # x0.shape  [-1, 3, 3, 5]
    # x1.shape  [-1, 3, 3, 5]
    # x2.shape  [-1, 3, 3, 5]
    
    x0, x1, x2 = fluid.layers.split(input, num_or_sections=[2, 3, 4], dim=2)
    # x0.shape  [-1, 3, 2, 5]
    # x1.shape  [-1, 3, 3, 5]
    # x2.shape  [-1, 3, 4, 5]









