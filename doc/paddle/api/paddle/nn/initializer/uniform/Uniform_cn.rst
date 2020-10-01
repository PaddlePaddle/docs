.. _cn_api_nn_initializer_Uniform:

Uniform
-------------------------------

.. py:class:: paddle.fluid.initializer.Uniform(low=-1.0, high=1.0) 


随机均匀分布初始化函数。

参数：
    - **low** (float，可选) - 下界。默认值为 -1.0。
    - **high** (float，可选) - 上界。默认值为 1.0。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为None。

返回：
    由随机均匀分布初始化的参数。

**代码示例**

.. code-block:: python
       
    import paddle

    data = paddle.ones(shape=[3, 1, 2], dtype='float32')
    weight_attr = paddle.framework.ParamAttr(
        name="linear_weight",
        initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
    bias_attr = paddle.framework.ParamAttr(
        name="linear_bias",
        initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
    linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
    # linear.weight:  [[-0.46245047  0.05260676]
    #                  [ 0.38054508  0.29169726]]
    # linear.bias:  [-0.2734719   0.23939109]
    
    res = linear(data)
    # res:  [[[-0.3553773  0.5836951]]
    #        [[-0.3553773  0.5836951]]
    #        [[-0.3553773  0.5836951]]]
