.. _user_guide_use_numpy_array_as_train_data:

###############################
Take Numpy Array as Train Data
###############################

PaddlePaddle Fluid supports you to configure data layer with :code:`fluid.layers.data()` .
Then you can use Numpy Array or directly use Python to create C++
:code:`fluid.LoDTensor` , with :code:`Executor.run(feed=...)` feeding to
:code:`fluid.Executor` or :code:`fluid.ParallelExecutor` 。

Configuration of Data Layer
############################

With :code:`fluid.layers.data()` ,you can configure data layer in neural network. Details are as follows:

.. code-block:: python

   import paddle.fluid as fluid

   image = fluid.layers.data(name="image", shape=[3, 224, 224])
   label = fluid.layers.data(name="label", shape=[1], dtype="int64")

   # use image/label as layer input
   prediction = fluid.layers.fc(input=image, size=1000, act="softmax")
   loss = fluid.layers.cross_entropy(input=prediction, label=label)
   ...

In the code above, :code:`image` and :code:`label` are two input data layers created by :code:`fluid.layers.data` . :code:`image` is float data of :code:`[3, 224, 224]` ; :code:`label` is the int data of :code:`[1]` . Attentions need to be paid:

1. :code:`-1` is represented for the dimension of batch size by default in Fluid.And :code:`-1` is added to the first shape of :code:`shape` by default. Therefore in the code above,it would be alright to transfer numpy array of :code:`[32, 3, 224, 224]` to :code:`image` . If you want to customize the rank of batch_size,please set 如果想自定义batch size :code:`fluid.layers.data(append_batch_size=False)` .Please refer to the user guide in higher rank :ref:`user_guide_customize_batch_size_rank` .

2. Data type as category label in Fluid is :code:`int64` and the label starts from 0. About the supported data types,please refer to :ref:`user_guide_paddle_support_data_types`。

.. _user_guide_feed_data_to_executor:

Transfer Train Data to Executor
################################

Both :code:`Executor.run` and :code:`ParallelExecutor.run` receive a parameter :code:`feed` .
The parameter is a dict in Python.Its key is the name of data layer,such as :code:`image` in code above. And its value is correspondent with numpy array.

For example:

.. code-block:: python

   exe = fluid.Executor(fluid.CPUPlace())
   exe.run(feed={
      "image": numpy.random.random(size=(32, 3, 224, 224)).astype('float32'),
      "label": numpy.random.random(size=(32, 1)).astype('int64')
   })

Advanced Usage
###############

How to feed Sequence Data
--------------------------

Sequence data is a typical data type supported by PaddlePaddle Fluid. You can take :code:`LoDTensor` as input data type.

You needs to: 

1. Feed all data to be trained in a mini-batch.

2. Get the length of each sequence.

You can use :code:`fluid.create_lod_tensor` to create :code:`LoDTensor`。

It needs to set the sequence nested depth :code:`lod_level` at the feed of sequence information.

For example,if train data are sentences consisting of words, :code:`lod_level=1`; if train data are paragraphs which consists of sentences that consists of words, :code:`lod_level=2` .

For example:

.. code-block:: python

   sentence = fluid.layers.data(name="sentence", dtype="int64", shape=[1], lod_level=1)

   ...

   exe.run(feed={
     "sentence": create_lod_tensor(
       data=numpy.array([1, 3, 4, 5, 3, 6, 8], dtype='int64').reshape(-1, 1),
       lod=[4, 1, 2],
       place=fluid.CPUPlace()
     )
   })

Train data :code:`sentence` contains three samples, the lengths of which are :code:`4, 1, 2` respectively.

They are :code:`data[0:4]`, :code:`data[4:5]` and :code:`data[5:7]` respectively.

How to set train data of every device in ParallelExecutor
----------------------------------------------------------

When you feed data to :code:`ParallelExecutor.run(feed=...)` , 
you can explicitly assign data for every train device (such as GPU).

You need to feed a list to :code:`feed` . Each element of the list is a dict.

The key of dict is name of data layer and the value of dict is value of data layer.

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

In code above,GPU0 will train 32 samples and GPU1 will train 16 samples.

.. _user_guide_customize_batch_size_rank:

Customize the shape of BatchSize
------------------------------------

Batch size is the first shape of data by default in PaddlePaddle Fluid, indicated by :code:`-1` .But in advanced usage,batch_size could be fixed or respresented by other shapes or multiple shapes,which could be implemented by setting :code:`fluid.layers.data(append_batch_size=False)` .

1. batch size with fixed shape

  .. code-block:: python

     image = fluid.layers.data(name="image", shape=[32, 784], append_batch_size=False)

  Here:code:`image` is always a matrix with size of :code:`[32, 784]` .

2. batch size expressed by other shapes

  .. code-block:: python

     sentence = fluid.layers.data(name="sentence",
                                  shape=[80, -1, 1],
                                  append_batch_size=False,
                                  dtype="int64")

  Here the middle shape of :code:`sentence` is batch size. The arrangement of data is applied in fixed-length recurrent neural network.

.. _user_guide_paddle_support_data_types:

Data types supported by Fluid
-------------------------------

Data types supported by PaddlePaddle Fluid contains:

   * float16： support part operation
   * float32:  major real number
   * float64:  minor real number,support most operation
   * int32: minor label
   * int64: major label
   * uint64: minor label
   * bool:  control flow
   * int16: minor label
   * uint8: input data type, used for pixel of picture