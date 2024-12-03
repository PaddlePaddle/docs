.. _api_guide_data_feeder_en:

Feed training/inference data with DataFeeder
########################################################

Fluid provides the :code:`DataFeeder` class, which converts data types such as numpy array into a :code:`DenseTensor` type to feed the training/inference network.

To create a :code:`DataFeeder` object:

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.layers.data(name='image', shape=[-1, 3, 224, 224], dtype='float32')
    label = fluid.layers.data(name='label', shape=[-1, 1], dtype='int64')
    place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
    feeder = fluid.DataFeeder(feed_list=[image, label], place=place)

The :code:`feed_list` parameter is a list of variables created by :code:`fluid.layers.data()` .
The :code:`place` parameter indicates that data such as numpy array passed in from the Python side should be converted to GPU or CPU :code:`DenseTensor`.
After creating the :code:`DataFeeder` object, the user can call the :code:`feed(iterable)` method to convert :code:`iterable` data given by user into :code:`DenseTensor` .

:code:`iterable` should be a object of Python List or a Tuple type, and each element in :code:`iterable` is a Python List of length N or Tuple type object, where N is the number of :code:`feed_list` variables passed in when the :code:`DataFeeder` object is created.

The concrete format of :code:`iterable` is:

.. code-block:: python

    iterable = [
        (image_1, label_1),
        (image_2, label_2),
        ...
        (image_n, label_n)
    ]

:code:`image_i` and :code:`label_i` are both numpy array data. If the dimension of the input data is [1], such as :code:`label_i`,
you can feed Python int, float, and other types of data. The data types and dimensions of :code:`image_i` and :code:`label_i` are not necessarily
the same as :code:`dtype` and :code:`shape` specified at :code:`fluid.layers.data()`. :code:`DataFeeder` internally
performs the conversion of data types and dimensions. If the :code:`lod_level` of the variable in :code:`feed_list` is not zero, in Fluid, the 0th dimension of each row in the dimensionally converted :code:`iterable` will be returned as :code:`LoD` .

Read :ref:`api_fluid_DataFeeder` for specific usage.
