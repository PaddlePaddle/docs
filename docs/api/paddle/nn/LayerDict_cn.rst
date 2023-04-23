.. _cn_api_nn_LayerDict:

LayerDict
-------------------------------

.. py:class:: paddle.nn.LayerDict(sublayers=None)




LayerDict 用于保存子层到有序字典中，它包含的子层将被正确地注册和添加。列表中的子层可以像常规 python 有序字典一样被访问。

参数
::::::::::::

    - **sublayers** (LayerDict|OrderedDict|list[(key, Layer)]，可选) - 键值对的可迭代对象，值的类型为 `paddle.nn.Layer` 。


代码示例
::::::::::::

.. code-block:: python

    import paddle
    from collections import OrderedDict

    sublayers = OrderedDict([
        ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
        ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
        ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
    ])

    layers_dict = paddle.nn.LayerDict(sublayers=sublayers)

    l = layers_dict['conv1d']

    for k in layers_dict:
        l = layers_dict[k]

    len(layers_dict)
    #3

    del layers_dict['conv2d']
    len(layers_dict)
    #2

    conv1d = layers_dict.pop('conv1d')
    len(layers_dict)
    #1

    layers_dict.clear()
    len(layers_dict)
    #0

方法
::::::::::::
clear()
'''''''''

清除 LayerDict 中所有的子层。

**参数**

    无。

**代码示例**

.. code-block:: python

    import paddle
    from collections import OrderedDict

    sublayers = OrderedDict([
        ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
        ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
        ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
    ])

    layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
    len(layer_dict)
    #3

    layer_dict.clear()
    len(layer_dict)
    #0

pop()
'''''''''

移除 LayerDict 中的键 并且返回该键对应的子层。

**参数**

    - **key** (str) - 要移除的 key。

**代码示例**

.. code-block:: python

    import paddle
    from collections import OrderedDict

    sublayers = OrderedDict([
        ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
        ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
        ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
    ])

    layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
    len(layer_dict)
    #3

    layer_dict.pop('conv2d')
    len(layer_dict)
    #2

keys()
'''''''''

返回 LayerDict 中键的可迭代对象。

**参数**

    无。

**代码示例**

.. code-block:: python

    import paddle
    from collections import OrderedDict

    sublayers = OrderedDict([
        ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
        ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
        ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
    ])

    layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
    for k in layer_dict.keys():
        print(k)

    #conv1d
    #conv2d
    #conv3d


items()
'''''''''

返回 LayerDict 中键/值对的可迭代对象。

**参数**

    无。

**代码示例**

.. code-block:: python

    import paddle
    from collections import OrderedDict

    sublayers = OrderedDict([
        ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
        ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
        ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
    ])

    layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
    for k, v in layer_dict.items():
        print(k, ":", v)

    #conv1d : Conv1D(3, 2, kernel_size=[3], data_format=NCL)
    #conv2d : Conv2D(3, 2, kernel_size=[3, 3], data_format=NCHW)
    #conv3d : Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)


values()
'''''''''

返回 LayerDict 中值的可迭代对象。

**参数**

    无。

**代码示例**

.. code-block:: python

    import paddle
    from collections import OrderedDict

    sublayers = OrderedDict([
        ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
        ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
        ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
    ])

    layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
    for v in layer_dict.values():
        print(v)

    #Conv1D(3, 2, kernel_size=[3], data_format=NCL)
    #Conv2D(3, 2, kernel_size=[3, 3], data_format=NCHW)
    #Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)


update()
'''''''''

更新子层中的键/值对到 LayerDict 中，会覆盖已经存在的键。

**参数**

    - **sublayers** (LayerDict|OrderedDict|list[(key, Layer)]) - 键值对的可迭代对象，值的类型为 `paddle.nn.Layer` 。

**代码示例**

.. code-block:: python

    import paddle
    from collections import OrderedDict

    sublayers = OrderedDict([
        ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
        ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
        ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
    ])

    new_sublayers = OrderedDict([
        ('relu', paddle.nn.ReLU()),
        ('conv2d', paddle.nn.Conv2D(4, 2, 4)),
    ])
    layer_dict = paddle.nn.LayerDict(sublayers=sublayers)

    layer_dict.update(new_sublayers)

    for k, v in layer_dict.items():
        print(k, ":", v)
    #conv1d : Conv1D(3, 2, kernel_size=[3], data_format=NCL)
    #conv2d : Conv2D(4, 2, kernel_size=[4, 4], data_format=NCHW)
    #conv3d : Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)
    #relu : ReLU()
