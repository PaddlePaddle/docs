.. _cn_api_fluid_layers_expand_as:

expand_as
-------------------------------

.. py:function:: paddle.fluid.layers.expand_as(x, target_tensor, name=None)

:alias_main: paddle.expand_as
:alias: paddle.expand_as,paddle.tensor.expand_as,paddle.tensor.manipulation.expand_as
:old_api: paddle.fluid.layers.expand_as



该OP会根据输入的variable ``target_tensor`` 对输入 ``x`` 的各维度进行广播。通过 ``target_tensor``的维度来为 ``x`` 的每个维度设置广播的次数，使得x 的维度与target_tensor的维度相同。 ``x`` 的秩应小于等于6。注意， ``target_tensor`` 的秩必须与 ``x`` 的秩相同。
注意:``target_tensor`` 对应的每一维必须能整除输入x中对应的维度，否则会报错。比如，target_tensor的维度为[2,6,2],x为[2,3,1],则整除后为[1,2,2]，x广播后维度为[2,6,2]。如果target_tensor的维度为[2,5,2]，第二维5不能整除x的第二维3，则会报错。        

以下是一个示例：

::

        输入(x) 是一个形状为[2, 3, 1]的 3-D Tensor :

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        target_tensor的维度 :  [2, 6, 2]

        输出(out) 是一个形状为[2, 6, 2]的 3-D Tensor:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]
                
        

参数:
        - **x** （Variable）- 维度最高为6的多维 ``Tensor`` 或 ``LoDTensor``，数据类型为 ``float32``，``float64``，``int32`` 或 ``bool``。
        - **target_tensor** （list|tuple|Variable）- 数据类型为 ``float32``，``float64``，``int32`` 或 ``bool`` 。可为Tensor或者LODTensor。
        - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置。默认值： ``None``。

返回：维度与输入 ``x`` 相同的 ``Tensor`` 或 ``LoDTensor``，数据类型与 ``x`` 相同。返回值的每个维度的大小等于``target_tensor`` 对应的维度的大小。

返回类型：``Variable`` 。

抛出异常：
    - :code:`ValueError`：``target_tensor`` 对应的每一维必须能整除输入x中对应的维度，否则会报错。

**代码示例**

..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        data = fluid.data(name="data", shape=[-1,10], dtype='float64')
        target_tensor = fluid.data(name="target_tensor", shape=[-1,20], dtype='float64')
        result = fluid.layers.expand_as(x=data, target_tensor=target_tensor) 
        use_cuda = False
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        x = np.random.rand(3,10)
        y = np.random.rand(3,20)
        output= exe.run(feed={"data":x,"target_tensor":y},fetch_list=[result.name])
        print(output[0].shape)
        #(3,20)