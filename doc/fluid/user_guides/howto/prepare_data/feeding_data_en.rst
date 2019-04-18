.. _user_guide_use_numpy_array_as_train_data_en:

#################################
Take Numpy Array as Training Data
#################################

PaddlePaddle Fluid supports configuring data layer with :code:`fluid.layers.data()` .
Then you can use Numpy Array or directly use Python to create C++
:code:`fluid.LoDTensor` , and then feed it to :code:`fluid.Executor` or :code:`fluid.ParallelExecutor` 
through :code:`Executor.run(feed=...)` .

Configure Data Layer
############################

With :code:`fluid.layers.data()` , you can configure data layer in neural network. Details are as follows:

.. code-block:: python

   import paddle.fluid as fluid

   image = fluid.layers.data(name="image", shape=[3, 224, 224])
   label = fluid.layers.data(name="label", shape=[1], dtype="int64")

   # use image/label as layer input
   prediction = fluid.layers.fc(input=image, size=1000, act="softmax")
   loss = fluid.layers.cross_entropy(input=prediction, label=label)
   ...

In the code above, :code:`image` and :code:`label` are two input data layers created by :code:`fluid.layers.data` . :code:`image` is float data of shape :code:`[3, 224, 224]` ; :code:`label` is the int data of shape :code:`[1]` . Note that:

1. :code:`-1` is represented for the dimension of batch size by default in Fluid. And :code:`-1` is added to the first dimension of :code:`shape` by default. Therefore in the code above, it would be alright to transfer numpy array of :code:`[32, 3, 224, 224]` to :code:`image` . If you want to customize the position of the batch size dimension, please set :code:`fluid.layers.data(append_batch_size=False)` .Please refer to the tutorial in the advanced user guide: :ref:`user_guide_customize_batch_size_rank_en` .

2. Data type of category labels in Fluid is :code:`int64` and the label starts from 0. About the supported data types,please refer to :ref:`user_guide_paddle_support_data_types_en` .

.. _user_guide_feed_data_to_executor_en:

Transfer Train Data to Executor
################################

Both :code:`Executor.run` and :code:`ParallelExecutor.run` receive a parameter :code:`feed` .
The parameter is a dict in Python. Its key is the name of data layer,such as :code:`image` in code above. And its value is the corresponding  numpy array.

For example:

.. code-block:: python

   exe = fluid.Executor(fluid.CPUPlace())
   # init Program
   exe.run(fluid.default_startup_program())
   exe.run(feed={
      "image": numpy.random.random(size=(32, 3, 224, 224)).astype('float32'),
      "label": numpy.random.random(size=(32, 1)).astype('int64')
   })

Advanced Usage
###############

How to feed Sequence Data
--------------------------

Sequence data is a unique data type supported by PaddlePaddle Fluid. You can take :code:`LoDTensor` as input data type.

You need to: 

1. Feed all data to be trained in a mini-batch.

2. Get the length of each sequence.

You can use :code:`fluid.create_lod_tensor` to create :code:`LoDTensor` .

To feed sequence information, it is necessary to set the sequence nested depth :code:`lod_level` .

For instance, if the training data are sentences consisting of words, :code:`lod_level=1`; if train data are paragraphs which consists of sentences that consists of words, :code:`lod_level=2` .

For example:

.. code-block:: python

   sentence = fluid.layers.data(name="sentence", dtype="int64", shape=[1], lod_level=1)

   ...

   exe.run(feed={
     "sentence": create_lod_tensor(
       data=numpy.array([1, 3, 4, 5, 3, 6, 8], dtype='int64').reshape(-1, 1),
       recursive_seq_lens=[[4, 1, 2]],
       place=fluid.CPUPlace()
     )
   })

Training data :code:`sentence` contain three samples, the lengths of which are :code:`4, 1, 2` respectively.

They are :code:`data[0:4]`, :code:`data[4:5]` and :code:`data[5:7]` respectively.

How to prepare training data for every device in ParallelExecutor
-------------------------------------------------------------------

When you feed data to :code:`ParallelExecutor.run(feed=...)` , 
you can explicitly assign data for every training device (such as GPU).

You need to feed a list to :code:`feed` . Each element of the list is a dict.

The key of the dict is name of data layer and the value of dict is value of data layer.

For example:

.. code-block:: python

   parallel_executor = fluid.ParallelExecutor()
   parallel_executor.run(
     feed=[
        {
          "image": numpy.random.random(size=(32, 3, 224, 224)).astype('float32'),
          "label": numpy.random.random(size=(32, 1)).astype('int64')
        },
        {
          "image": numpy.random.random(size=(16, 3, 224, 224)).astype('float32'),
          "label": numpy.random.random(size=(16, 1)).astype('int64')
        },
     ]
   )

In the code above, GPU0 will train 32 samples and GPU1 will train 16 samples.

.. _user_guide_customize_batch_size_rank_en:

Customize the BatchSize dimension
------------------------------------

Batch size is the first dimension of data by default in PaddlePaddle Fluid, indicated by :code:`-1` .But in advanced usage, batch_size could be fixed or respresented by other dimension or multiple dimensions, which could be implemented by setting :code:`fluid.layers.data(append_batch_size=False)` .

1. fixed BatchSize dimension

  .. code-block:: python

     image = fluid.layers.data(name="image", shape=[32, 784], append_batch_size=False)

  Here :code:`image` is always a matrix with size of :code:`[32, 784]` .

2. batch size expressed by other dimension

  .. code-block:: python

     sentence = fluid.layers.data(name="sentence",
                                  shape=[80, -1, 1],
                                  append_batch_size=False,
                                  dtype="int64")

  Here the middle dimension of :code:`sentence` is batch size. This type of data layout is applied in fixed-length recurrent neural networks.

.. _user_guide_paddle_support_data_types_en:

Data types supported by Fluid
-------------------------------

Data types supported by PaddlePaddle Fluid contains:

   * float16: supported by part of operations
   * float32: major data type of real number
   * float64: minor data type of real number, supported by most operations
   * int32: minor data type of labels
   * int64: major data type of labels
   * uint64: minor data type of labels
   * bool:  type of control flow data
   * int16: minor type of labels
   * uint8: input data type, used for pixel of picture
