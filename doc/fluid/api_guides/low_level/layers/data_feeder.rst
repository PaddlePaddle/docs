..  _api_guide_data_feeder:

使用DataFeeder传入训练/预测数据
###################################

Fluid提供 :code:`DataFeeder` 类，将numpy array等数据转换为 :code:`LoDTensor` 类型传入训练/预测网络。

用户创建 :code:`DataFeeder` 对象的方式为：

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.layers.data(name='image', shape=[-1, 3, 224, 224], dtype='float32')
    label = fluid.layers.data(name='label', shape=[-1, 1], dtype='int64')
    place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[image, label], place=place)

其中，:code:`feed_list` 参数为变量列表，这些变量由 :code:`fluid.layers.data()` 创建，
:code:`place` 参数表示应将Python端传入的numpy array等数据转换为GPU端或是CPU端的 :code:`LoDTensor` 。
创建 :code:`DataFeeder` 对象后，用户可调用其 :code:`feed(iterable)` 方法将用户传入的
:code:`iterable` 数据转换为 :code:`LoDTensor`。

:code:`iterable` 应为Python List或Tuple类型对象，且 :code:`iterable` 的每个元素均为长度为N的
Python List或Tuple类型对象，其中N为创建 :code:`DataFeeder` 对象时传入的 :code:`feed_list` 变量个数。

:code:`iterable` 的具体格式为：

.. code-block:: python

    iterable = [
        (image_1, label_1),
        (image_2, label_2),
        ...
        (image_n, label_n)
    ]

其中，:code:`image_i` 与 :code:`label_i` 均为numpy array类型数据。若传入数据的维度为[1]，如 :code:`label_i`,
则可传入Python int、float等类型数据。 :code:`image_i` 与 :code:`label_i` 的数据类型和维度不必
与 :code:`fluid.layers.data()` 创建时指定的 :code:`dtype` 和 :code:`shape` 完全一致，:code:`DataFeeder` 内部
会完成数据类型和维度的转换。若 :code:`feed_list` 中的变量的 :code:`lod_level` 不为零，则Fluid会将经过维度转换后的
:code:`iterable` 中每行数据的第0维作为返回结果的 :code:`LoD`。

具体使用方法请参见 :ref:`cn_api_fluid_DataFeeder` 。