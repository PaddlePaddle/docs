.. _cn_api_nn_initializer_Uniform:

Uniform
-------------------------------

.. py:class:: paddle.fluid.initializer.Uniform(low=-1.0, high=1.0) 




随机均匀分布初始化器

参数：
    - **low** (float16|float32) - 下界 
    - **high** (float16|float32) - 上界
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回：对象

**代码示例**

.. code-block:: python
       
    import paddle

    data = paddle.ones(shape=[3, 1, 2], dtype='float32')
    weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
        trainable=False, regularizer=None, initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
    bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
        trainable=False, regularizer=None, initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
    linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
    res = linear(data)
 







