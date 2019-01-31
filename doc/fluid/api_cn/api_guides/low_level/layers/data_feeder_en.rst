.. _api_guide_data_feeder_en:

Feed training/predictive data with DataFeeder
########################################################

Fluid provides the :code:`DataFeeder` class, which converts data such as numpy array into a :code:`LoDTensor` type into the training/prediction network.

The way the user creates the :code:`DataFeeder` object is:

.. code-block:: python

    Import paddle.fluid as fluid

    Image = fluid.layers.data(name='image', shape=[-1, 3, 224, 224], dtype='float32')
    Label = fluid.layers.data(name='label', shape=[-1, 1], dtype='int64')
    Place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
    Feeder = fluid.DataFeeder(feed_list=[image, label], place=place)

Where the :code:`feed_list` parameter is a list of variables created by :code:`fluid.layers.data()`
The :code:`place` parameter indicates that data such as numpy array passed in from the Python side should be converted to GPU or CPU: code:`LoDTensor`.
After creating the :code:`DataFeeder` object, the user can call the :code:`feed(iterable)` method to convert :code:`iterable` user pass in to :code:`LoDTensor` .

:code:`iterable` should be a Python List or a Tuple type object, and each element of :code:`iterable` is of length N
A Python List or Tuple type object, where N is the number of :code:`feed_list` variables passed in when the :code:`DataFeeder` object is created.

The specific format of :code:`iterable` is:

.. code-block:: python

    Iterable = [
        (image_1, label_1),
        (image_2, label_2),
        ...
        (image_n, label_n)
    ]

Among them, :code:`image_i` and :code:`label_i` are both numpy array type data. If the dimension of the incoming data is [1], such as :code:`label_i`,
You can pass in Python int, float, and other types of data. The data types and dimensions of :code:`image_i` and :code:`label_i` are not necessary
Same as :code:`dtype` and :code:`shape` specified at :code:`fluid.layers.data()`, :code:`DataFeeder` internal
The conversion of data types and dimensions is completed. If the :code:`lod_level` of the variable in :code:`feed_list` is not zero, in Fluid,  0th dimension of each line of data dimensionally converted in :code:`iterable` will be returned as :code:`LoD` in result.

See :ref:`api_fluid_DataFeeder` for specific usage.