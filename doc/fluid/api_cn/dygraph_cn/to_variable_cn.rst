.. _cn_api_fluid_dygraph_to_variable:

to_variable
-------------------------------

.. py:function:: paddle.fluid.dygraph_to_variable(value, block=None, name=None)

该函数会从ndarray创建一个variable。

参数：
    - **value**  (ndarray) – 需要转换的numpy值
    - **block**  (fluid.Block) – variable所在的block，默认为None
    - **name**  (str) – variable的名称，默认为None


返回： 从指定numpy创建的variable

返回类型：Variable

**代码示例**:

.. code-block:: python
    
    import numpy as np
    import paddle.fluid as fluid

    with fluid.dygraph.guard():
        x = np.ones([2, 2], np.float32)
        y = fluid.dygraph.to_variable(x)






