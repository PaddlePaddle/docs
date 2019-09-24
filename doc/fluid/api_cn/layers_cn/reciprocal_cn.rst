.. _cn_api_fluid_layers_reciprocal:

reciprocal
-------------------------------

.. py:function:: paddle.fluid.layers.reciprocal(x, name=None)

reciprocal（对输入取倒数）激活函数


.. math::
    out = \frac{1}{x}

参数:

    - **x** - 输入的多维Tensor,支持的数据类型为float32，float64。
    - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。


返回： 对输入取倒数得到的Tensor，输出Tensor数据类型与输入相同，输出Tensor维度和输入维度相同。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.reciprocal(data)












