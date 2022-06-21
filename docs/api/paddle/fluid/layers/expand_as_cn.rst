.. _cn_api_fluid_layers_expand_as:

expand_as
-------------------------------

.. py:function:: paddle.fluid.layers.expand_as(x, target_tensor, name=None)




该OP会根据输入的variable ``target_tensor`` 对输入 ``x`` 的各维度进行广播。通过 ``target_tensor``的维度来为 ``x`` 的每个维度设置广播的次数，使得x 的维度与target_tensor的维度相同。``x`` 的秩应小于等于6。注意，``target_tensor`` 的秩必须与 ``x`` 的秩相同。
注意：``target_tensor`` 对应的每一维必须能整除输入x中对应的维度，否则会报错。比如，target_tensor的维度为[2,6,2],x为[2,3,1]，则整除后为[1,2,2]，x广播后维度为[2,6,2]。如果target_tensor的维度为[2,5,2]，第二维5不能整除x的第二维3，则会报错。

以下是一个示例：

::

        输入(x) 是一个形状为[2, 3, 1]的 3-D Tensor :

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        target_tensor的维度：[2, 6, 2]

        输出(out) 是一个形状为[2, 6, 2]的 3-D Tensor:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]
                
        

参数
::::::::::::

        - **x** （Variable）- 维度最高为6的多维 ``Tensor`` 或 ``LoDTensor``，数据类型为 ``float32``，``float64``，``int32`` 或 ``bool``。
        - **target_tensor** （list|tuple|Variable）- 数据类型为 ``float32``，``float64``，``int32`` 或 ``bool``。可为Tensor或者LODTensor。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
维度与输入 ``x`` 相同的 ``Tensor`` 或 ``LoDTensor``，数据类型与 ``x`` 相同。返回值的每个维度的大小等于``target_tensor`` 对应的维度的大小。

返回类型
::::::::::::
``Variable`` 。

抛出异常
::::::::::::

    - :code:`ValueError`：``target_tensor`` 对应的每一维必须能整除输入x中对应的维度，否则会报错。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.expand_as