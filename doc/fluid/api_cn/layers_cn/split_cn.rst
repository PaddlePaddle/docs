.. _cn_api_fluid_layers_split:

split
-------------------------------

.. py:function:: paddle.fluid.layers.split(input,num_or_sections,dim=-1,name=None)

将输入Tensor分割成多个子Tensor。

参数：
    - **input** (Variable) - 输入变量，为数据类型为int32，int64，float32，float64的多维Tensor或者LoDTensor。
    - **num_or_sections** (int|list) - 整数或元素为整数的列表。如果\ ``num_or_sections``\ 是一个整数，则表示Tensor平均划分为的相同大小子Tensor的数量。如果\ ``num_or_sections``\ 是一个整数列表，则列表的长度代表子Tensor的数量，列表中的整数依次代表子Tensor的需要分割成的维度的大小。列表长度不能超过输入Tensor待分割的维度的大小。
    - **dim** (int) - 需要分割的维度。如果dim < 0,划分的维度为rank(input) + dim，数据类型为int32，int64。
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：分割后的Tensor列表。

返回类型：列表(Variable(Tensor|LoDTensor))，数据类型为int32，int64，float32，float64。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid

    # 输入是维度为[-1, 3, 9, 5]的Tensor：
    input = fluid.layers.data(
         name="input", shape=[3, 9, 5], dtype="float32")

    # 传入num_or_sections为一个整数
    x0, x1, x2 = fluid.layers.split(input, num_or_sections=3, dim=2)
    x0.shape  # [-1, 3, 3, 5]
    x1.shape  # [-1, 3, 3, 5]
    x2.shape  # [-1, 3, 3, 5]

    # 传入num_or_sections为一个整数列表
    x0, x1, x2 = fluid.layers.split(input, num_or_sections=[2, 3, 4], dim=2)
    x0.shape  # [-1, 3, 2, 5]
    x1.shape  # [-1, 3, 3, 5]
    x2.shape  # [-1, 3, 4, 5]









