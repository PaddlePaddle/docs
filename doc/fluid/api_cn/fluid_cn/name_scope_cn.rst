.. _cn_api_fluid_name_scope:

name_scope
-------------------------------

.. py:function:: paddle.fluid.name_scope(prefix=None)


为operators生成层次名称前缀

注意： 这个函数只能用于调试和可视化。不要将其用于分析，比如graph/program转换。

参数：
  - **prefix** (str) - 前缀

**示例代码**

.. code-block:: python
          
     import paddle.fluid as fluid
     with fluid.name_scope("s1"):
        a = fluid.layers.data(name='data', shape=[1], dtype='int32')
        b = a + 1
        with fluid.name_scope("s2"):
           c = b * 1
        with fluid.name_scope("s3"):
           d = c / 1
     with fluid.name_scope("s1"):
           f = fluid.layers.pow(d, 2.0)
     with fluid.name_scope("s4"):
           g = f - 1



