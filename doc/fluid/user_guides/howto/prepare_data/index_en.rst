..  _user_guide_prepare_data_en:

#############
Prepare Data
#############

PaddlePaddle Fluid supports two methods to feed data into networks:

1. Synchronous method - Python Reader：Firstly, use :code:`fluid.layers.data` to set up data input layer. Then, feed in the training data through :code:`executor.run(feed=...)` in :code:`fluid.Executor` or :code:`fluid.ParallelExecutor` .

2. Asynchronous method - py_reader：Firstly, use :code:`fluid.layers.py_reader` to set up data input layer. Then configure the data source with functions :code:`decorate_paddle_reader` or :code:`decorate_tensor_provider` of :code:`py_reader` . After that, call :code:`fluid.layers.read_file` to read data.



Comparisons of the two methods:

=========================  ====================================================   ===============================================
Aspects                                   Synchronous Python Reader                       Asynchronous py_reader
=========================  ====================================================   ===============================================
API interface                          :code:`executor.run(feed=...)`                 :code:`fluid.layers.py_reader`
data type                                   Numpy Array                                Numpy Array or LoDTensor
data augmentation          carried out by other libraries on Python end            carried out by other libraries on Python end 
velocity                                        slow                                            rapid
recommended applications                model debugging                                      industrial training
=========================  ====================================================   ===============================================

Synchronous Python Reader
##########################

Fluid provides Python Reader to feed in data.

Python Reader is a pure Python-side interface, and data feeding is synchronized with the model training/prediction process. Users can pass in data through Numpy Array. For specific operations, please refer to:


.. toctree::
   :maxdepth: 1

   feeding_data_en.rst

Python Reader supports advanced functions like group batch, shuffle. For specific operations, please refer to：

.. toctree::
   :maxdepth: 1

   reader.md

Asynchronous py_reader
########################

Fluid provides asynchronous data feeding method PyReader. It is more efficient as data feeding is not synchronized with the model training/prediction process. For specific operations, please refer to：

.. toctree::
   :maxdepth: 1

   use_py_reader_en.rst
