.. _cn_api_fluid_dygraph_ParameterList:

ParameterList
-------------------------------

.. py:class:: paddle.fluid.dygraph.ParameterList(parameters=None)




参数列表容器。此容器的行为类似于Python列表，但它包含的参数将被正确地注册和添加。

参数：
    - **parameters** (iterable，可选) - 可迭代的Parameters。

返回：无

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    class MyLayer(fluid.Layer):
        def __init__(self, num_stacked_param):
            super(MyLayer, self).__init__()
            # 使用可迭代的 Parameters 创建 ParameterList
            self.params = fluid.dygraph.ParameterList(
                [fluid.layers.create_parameter(
                    shape=[2, 2], dtype='float32')] * num_stacked_param)
        def forward(self, x):
            for i, p in enumerate(self.params):
                tmp = self._helper.create_variable_for_type_inference('float32')
                self._helper.append_op(
                    type="mul",
                    inputs={"X": x,
                            "Y": p},
                    outputs={"Out": tmp},
                    attrs={"x_num_col_dims": 1,
                           "y_num_col_dims": 1})
                x = tmp
            return x

    data_np = np.random.uniform(-1, 1, [5, 2]).astype('float32')
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(data_np)
        num_stacked_param = 4
        model = MyLayer(num_stacked_param)
        print(len(model.params))  # 4
        res = model(x)
        print(res.shape)  # [5, 2]
        replaced_param = fluid.layers.create_parameter(shape=[2, 3], dtype='float32')
        model.params[num_stacked_param - 1] = replaced_param  # 替换最后一个参数
        res = model(x)
        print(res.shape)  # [5, 3]
        model.params.append(fluid.layers.create_parameter(shape=[3, 4], dtype='float32'))  # 添加参数
        print(len(model.params))  # 5
        res = model(x)
        print(res.shape)  # [5, 4]


