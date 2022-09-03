.. _api_guide_data_in_out_en:

Data input and output
######################


Data input
-------------

Fluid supports two methods for data input, including:

1. Python Reader: A pure Python Reader. The user defines the :code:`fluid.layers.data` layer on the Python side and builds the network.
Then, read the data by calling :code:`executor.run(feed=...)` . The process of data reading and model training/inference is performed simultaneously.

2. PyReader: An Efficient and flexible C++ Reader interface. PyReader internally maintains a queue with size of :code:`capacity`  (queue capacity is determined by
:code:`capacity` parameter in the :code:`fluid.layers.py_reader` interface ). Python side call queue :code:`push` to feed the training/inference data, and the C++ side training/inference program calls the :code:`pop` method to retrieve the data sent by the Python side. PyReader can work in conjunction with :code:`double_buffer` to realize asynchronous execution of data reading and model training/inference.

For details, please refer to :ref:`api_fluid_layers_py_reader`.


Data output
------------

Fluid supports obtaining data for the current batch in the training/inference phase.

The user can fetch expected variables from :code:`executor.run(fetch_list=[...], return_numpy=...)` . User can determine whether to convert the output data to numpy array by setting the :code:`return_numpy` parameter.
If :code:`return_numpy` is :code:`False` , data of type :code:`LoDTensor` will be returned.

For specific usage, please refer to the relevant API documentation :ref:`api_fluid_executor_Executor` and
:ref:`api_fluid_ParallelExecutor`.
