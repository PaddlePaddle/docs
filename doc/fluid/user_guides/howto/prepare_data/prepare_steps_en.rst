..  _user_guide_prepare_steps_en:

#############
Prepare Steps
#############

Data preparation in PaddlePaddle Fluid can be separated into 3 steps.

Step 1: Define a reader to generate training/testing data
##########################################################

The generated data type can be Numpy Array or LoDTensor. According to the different data formats returned by the reader, it can be divided into Batch Reader and Sample Reader.

The batch reader yields a mini-batch data for each, while the sample reader yields a sample data for each.

If your reader yields a sample data, we provide a data augmentation and batching tool for you: :code:`Python Reader` .

Step 2: Define data layer variables in network
###############################################

Users should use :code:`fluid.data` to define data layer variables. Name, dtype and shape are required when defining. For example,

.. code-block:: python

    import paddle.fluid as fluid

    image = fluid.data(name='image', dtype='float32', shape=[None, 28, 28])
    label = fluid.data(name='label', dtype='int64', shape=[None, 1])

None means that the dimension is uncertain. In this example, None means the batch size.

Step 3: Send the data to network for training/testing
######################################################

PaddlePaddle Fluid provides 2 methods for sending data to the network: Asynchronous DataLoader API, and Synchronous Feed Method.

- Asynchronous DataLoader API

User should use :code:`fluid.io.DataLoader` to define a DataLoader object and use its setter method to set the data source.
When using DataLoader API, the process of data sending works asynchronously with network training/testing.
It is an efficient way for sending data and recommended to use.

- Synchronous Feed Method

User should create the feeding data beforehand and use :code:`executor.run(feed=...)` to send the data to :code:`fluid.Executor` or :code:`fluid.ParallelExecutor` .
Data preparation and network training/testing work synchronously, which is less efficient.

Comparison of these 2 methods are as follows:

==========================  =================================   ======================================
Comparison item                 Synchronous Feed Method              Asynchronous DataLoader API
==========================  =================================   ======================================
API                           :code:`executor.run(feed=...)`          :code:`fluid.io.DataLoader`
Data type                       Numpy Array or LoDTensor                Numpy Array or LoDTensor
Data augmentation            use Python for data augmentation       use Python for data augmentation
Speed                                     slow                                    rapid
Recommended applications            model debugging                        industrial training
==========================  =================================   ======================================

Choose different usages for different data formats
###################################################

According to the different data formats of reader, users should choose different usages for data preparation.

Read data from sample reader
+++++++++++++++++++++++++++++

If user-defined reader is a sample reader, users should use the following steps:

Step 1. Batching
=================

Use the data reader interfaces in PaddlePaddle Fluid for data augmentation and batching. Please refer to `Python Reader <./reader.html>`_ for details.

Step 2. Sending data
=====================

If using Asynchronous DataLoader API, please use :code:`set_sample_generator` or :code:`set_sample_list_generator` to set the data source for DataLoader. Please refer to :ref:`user_guide_use_py_reader_en` for details.

If using Synchronous Feed Method, please use DataFeeder to convert the reader data to LoDTensor before sending to the network. Please refer to :ref:`api_fluid_DataFeeder` for details.

Read data from sample reader
+++++++++++++++++++++++++++++

Step 1. Batching
=================

Since the reader has been a batch reader, this step can be skipped.

Step 2. Sending data
=====================

If using Asynchronous DataLoader API, please use :code:`set_batch_generator` to set the data source for DataLoader. Please refer to :ref:`user_guide_use_py_reader_en` for details.

If using Synchronous Feed Method, please refer to :ref:`user_guide_use_numpy_array_as_train_data_en` for details.