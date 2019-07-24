.. _cn_api_fluid_layers_lod_reset:

lod_reset
-------------------------------

.. py:function:: paddle.fluid.layers.lod_reset(x, y=None, target_lod=None)


设定x的LoD为y或者target_lod。如果提供y，首先将y.lod指定为目标LoD,否则y.data将指定为目标LoD。如果未提供y，目标LoD则指定为target_lod。如果目标LoD指定为Y.data或target_lod，只提供一层LoD。

::


    * 例1:

    给定一级LoDTensor x:
        x.lod =  [[ 2,           3,                   1 ]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    target_lod: [4, 2]

    得到一级LoDTensor:
        out.lod =  [[4,                          2]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

    * 例2:

    给定一级LoDTensor x:
        x.lod =  [[2,            3,                   1]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    y是张量（Tensor）:
        y.data = [[2, 4]]
        y.dims = [1, 3]

    得到一级LoDTensor:
        out.lod =  [[2,            4]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

    * 例3:

    给定一级LoDTensor x:
        x.lod =  [[2,            3,                   1]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    y是二级LoDTensor:
        y.lod =  [[2, 2], [2, 2, 1, 1]]
        y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
        y.dims = [6, 1]

    得到一个二级LoDTensor:
        out.lod =  [[2, 2], [2, 2, 1, 1]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

参数：
    - **x** (Variable)-输入变量，可以为Tensor或者LodTensor
    - **y** (Variable|None)-若提供，输出的LoD则衍生自y
    - **target_lod** (list|tuple|None)-一层LoD，y未提供时作为目标LoD

返回：输出变量，该层指定为LoD

返回类型：变量

抛出异常：``TypeError`` - 如果y和target_lod都为空

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[10])
    y = fluid.layers.data(name='y', shape=[10, 20], lod_level=2)
    out = fluid.layers.lod_reset(x=x, y=y)









