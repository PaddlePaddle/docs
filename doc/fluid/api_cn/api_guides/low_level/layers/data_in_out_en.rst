.. _api_guide_data_in_out_en:

Data input and output
######################


data input
-------------

Fluid supports two methods for data input, including:

1. Python Reader: A pure Python Reader. The user defines the :code:`fluid.layers.data` layer on the Python side and builds the network.
:code:`executor.run(feed=...)` Read the data in the same way. The process of data reading and model training/prediction is performed simultaneously.

2. PyReader: Efficient and flexible C++ Reader interface. PyReader internal maintenance capacity is :code:`capacity` queue (queue capacity is determined by
:code:`fluid.layers.py_reader` interface :code:`capacity` parameter setting), Python side call queue :code:`push`
The method feeds the training/predictive data, and the C++-side training/predicting program calls the :code:`pop` method to retrieve the data sent by the Python side. PyReader can be used with
:code:`double_buffer` works in conjunction with asynchronous execution of data reading and training/prediction.

For details, please refer to :ref:`api_fluid_layers_py_reader`.


Data output
------------

Fluid supports obtaining data for the current batch during the training/prediction phase.

The user can pass the :code:`executor.run(fetch_list=[...], return_numpy=...)`
The output variable expected by fetch is set to convert the output data to numpy array by setting the :code:`return_numpy` parameter.
If :code:`return_numpy` is :code:`False` , then return :code:`LoDTensor` type data.

For specific usage, please refer to the relevant API documentation: ref:`api_fluid_executor_Executor` and
:ref:`api_fluid_ParallelExecutor`.