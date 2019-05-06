#####################
LoD-Tensor User Guide
#####################

LoD(Level-of-Detail) Tensor is a unique term in Fluid, which can be constructed by appending sequence information to Tensor. Data transferred in Fluid contain input, output and learnable parameters of the network, all of which are represented by LoD-Tensor.

With the help of this user guide, you will learn the design idea of LoD-Tensor in Fluid so that you can use such a data type more flexibly.

Challenge of variable-length sequences
======================================

In most deep learning frameworks, a mini-batch is represented by Tensor.

For example, if there are 10 pictures in a mini-batch and the size of each picture is 32*32, the mini-batch will be a 10*32*32 Tensor.

Or in the NLP task, there are N sentences in a mini-batch and the length of each sentence is L. Every word is represented by a one-hot vector with D dimensions. Then the mini-batch can be represented by an N*L*D Tensor.

In the two examples above, the size of each sequence element remains the same. However, the data to be trained are variable-length sequences in many cases. For this scenario, method to be taken in most frameworks is to set a fixed length and sequence data shorter than the fixed length will be padded with 0 to reach the fixed length.

Owing to the LoD-Tensor in Fluid, it is not necessary to keep the lengths of sequence data in every mini-batch constant.Therefore tasks sensitive to sequence formats like NLP can also be finished without padding.

Index Data Structure (LoD) is introduced to Fluid to split Tensor into sequences.

Index Structure - LoD 
======================

To have a better understanding of the concept of LoD, you can refer to the examples in this section.

**mini-batch consisting of sentences**

Suppose a mini-batch contains three sentences, and each contains 3, 1, 2 words respectively. Then the mini-batch can be represented by a (3+1+2)*D Tensor with some index information appended:

.. code-block :: text

  3       1   2
  | | |   |   | |

In the text above, each :code:`|` represents a word vector with D dimension and a 1-level LoD is made up of digits 3,1,2 .

**recursive sequence**

Take a 2-level LoD-Tensor for example, a mini-batch contains articles of 3 sentences, 1 sentence and 2 sentences. The number of words in every sentence is different. Then the mini-batch is formed as follows:

.. code-block:: text


  3            1 2
  3   2  4     1 2  3
  ||| || ||||  | || |||


the LoD to express the format:

.. code-block:: text

  [[3，1，2]/*level=0*/，[3，2，4，1，2，3]/*level=1*/]


**mini-batch consisting of video data**

In the task of computer vision, it usually needs to deal objects with high dimension like videos and pictures. Suppose a mini-batch contains 3 videos, which is composed of 3 frames, 1 frames, 2 frames respectively. The size of each frame is 640*480. Then the mini-batch can be described as:

.. code-block:: text

  3     1  2
  口口口 口 口口


The size of the tensor at the bottom is (3+1+2)*640*480. Every :code:`口` represents a 640*480 picture.

**mini-batch consisting of pictures**

Traditionally, for a mini-batch of N pictures with fixed size, LoD-Tensor is described as:

.. code-block:: text

  1 1 1 1     1
  口口口口 ... 口

Under such circumstance, we will consider LoD-Tensor as a common tensor instead of ignoring information because of the indices of all elements are 1.

.. code-block:: text

  口口口口 ... 口

**model parameter**

model parameter is a common tensor which is described as a 0-level LoD-Tensor in Fluid.

LoDTensor expressed by offset
=============================

To have a quick access to the original sequence, you can take the offset expression method——store the first and last element of a sequence instead of its length.

In the example above, you can compute the length of fundamental elements:

.. code-block:: text

  3 2 4 1 2 3

It is expressed by offset as follows:

.. code-block:: text

  0  3  5   9   10  12   15
     =  =   =   =   =    =
     3  2+3 4+5 1+9 2+10 3+12

Therefore we infer that the first sentence starts from word 0 to word 3 and the second sentence starts from word 3 to word 5.

Similarly, for the length of the top layer of LoD

.. code-block:: text

  3 1 2

It can be expressed by offset:

.. code-block:: text

  0 3 4   6
    = =   =
    3 3+1 4+2

Therefore the LoD-Tensor is expressed by offset:

.. code-block:: text

  0       3    4      6
    3 5 9   10   12 15


LoD-Tensor
=============
A LoD-Tensor can be regarded as a tree of which the leaf is an original sequence element and branch is the flag of fundamental element.

There are two ways to express sequence information of LoD-Tensor in Fluid: primitive length and offset. LoD-Tensor is expressed by offset in Paddle to offer a quicker access to sequence;LoD-Tensor is expressed by primitive length in python API to make user understand and compute more easily. The primary length is named as  :code:`recursive_sequence_lengths` .

Take a 2-level LoD-Tensor mentioned above as an example:

.. code-block:: text

  3           1  2
  3   2  4    1  2  3
  ||| || |||| |  || |||

- LoD-Tensor expressed by offset: [ [0,3,4,6] , [0,3,5,9,10,12,15] ]
- LoD-Tensor expressed by primitive length: recursive_sequence_lengths=[ [3-0 , 4-3 , 6-4] , [3-0 , 5-3 , 9-5 , 10-9 , 12-10 , 15-12] ]


Take text sequence as an example,[3,1,2] indicates there are 3 articles in the mini-batch,which contains 3,1,2 sentences respectively.[3,2,4,1,2,3] indicates there are 3,2,4,1,2,3 words in sentences respectively.

recursive_seq_lens is a double Layer nested list, and in other words, the element of the list is list. The size of the outermost list represents the nested layers, namely the size of lod-level; Each inner list represents the size of each element in each lod-level. 

The following three pieces of codes introduce how to create LoD-Tensor, how to transform LoD-Tensor to Tensor and how to transform Tensor to LoD-Tensor respectively:

  * Create LoD-Tensor

.. code-block:: python

  #Create lod-tensor
  import paddle.fluid as fluid
  import numpy as np
  
  a = fluid.create_lod_tensor(np.array([[1],[1],[1],
                                    [1],[1],
                                    [1],[1],[1],[1],
                                    [1],
                                    [1],[1],
                                    [1],[1],[1]]).astype('int64') ,
                            [[3,1,2] , [3,2,4,1,2,3]],
                            fluid.CPUPlace())
  
  #Check lod-tensor nested layers
  print (len(a.recursive_sequence_lengths()))
  # output：2

  #Check the number of the most fundamental elements
  print (sum(a.recursive_sequence_lengths()[-1]))
  # output:15 (3+2+4+1+2+3=15)

* Transform LoD-Tensor to Tensor

  .. code-block:: python

   import paddle.fluid as fluid
   import numpy as np

   # create LoD-Tensor
   a = fluid.create_lod_tensor(np.array([[1.1], [2.2],[3.3],[4.4]]).astype('float32'), [[1,3]], fluid.CPUPlace())

   def LodTensor_to_Tensor(lod_tensor):
     # get lod information of LoD-Tensor
     lod = lod_tensor.lod()
     # transform into array
     array = np.array(lod_tensor)
     new_array = []
     # transform to Tensor according to the layer information of the original LoD-Tensor
     for i in range(len(lod[0]) - 1):
         new_array.append(array[lod[0][i]:lod[0][i + 1]])
     return new_array

   new_array = LodTensor_to_Tensor(a)

   # output the result
   print(new_array)

 * Transform Tensor to LoD-Tensor

  .. code-block:: python

   import paddle.fluid as fluid
   import numpy as np

   def to_lodtensor(data, place):
     # save the length of Tensor as LoD information
     seq_lens = [len(seq) for seq in data]
     cur_len = 0
     lod = [cur_len]
     for l in seq_lens:
         cur_len += l
         lod.append(cur_len)
     # decrease the dimention of transformed Tensor
     flattened_data = np.concatenate(data, axis=0).astype("int64")
     flattened_data = flattened_data.reshape([len(flattened_data), 1])
     # add lod information to Tensor data
     res = fluid.LoDTensor()
     res.set(flattened_data, place)
     res.set_lod([lod])
     return res

   # new_array is the transformed Tensor above
   lod_tensor = to_lodtensor(new_array,fluid.CPUPlace())

   # output LoD information
   print("The LoD of the result: {}.".format(lod_tensor.lod()))

   # examine the consistency with Tensor data
   print("The array : {}.".format(np.array(lod_tensor)))





Code examples
==============

Input variable x is expanded according to specified layer level y-lod in the code example in this section. The example below contains some fundamental conception of LoD-Tensor. By following the code, you will

-  Have a direct understanding of the implementation of :code:`fluid.layers.sequence_expand` in Fluid
-  Know how to create LoD-Tensor in Fluid
-  Learn how to print the content of LoDTensor


  
**Define the Process of Computing**

layers.sequence_expand expands x by obtaining the lod value of y. About more explanation of :code:`fluid.layers.sequence_expand` , please read :ref:`api_fluid_layers_sequence_expand` first. 

Code of sequence expanding:

.. code-block:: python

  x = fluid.layers.data(name='x', shape=[1], dtype='float32', lod_level=1)
  y = fluid.layers.data(name='y', shape=[1], dtype='float32', lod_level=2)
  out = fluid.layers.sequence_expand(x=x, y=y, ref_level=0)

*Note*：The dimension of input LoD-Tensor is only associated with the dimension of real data transferred in. The shape value set for x and y in the definition of network structure is just a placeholder with little influence on the result.  

**Create Executor**

.. code-block:: python

  place = fluid.CPUPlace()
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())

**Prepare Data**

Here we use :code:`fluid.create_lod_tensor` to create the input data of :code:`sequence_expand` and expand x_d by defining LoD of y_d. The output value is only associated with LoD of y_d. And the data of y_d is not invovled in the process of computation. The dimension of y_d must keep consistent with as its LoD[-1] .

About the user guide of :code:`fluid.create_lod_tensor()` , please refer to :ref:`api_fluid_create_lod_tensor` .

Code：

.. code-block:: python

  x_d = fluid.create_lod_tensor(np.array([[1.1],[2.2],[3.3],[4.4]]).astype('float32'), [[1,3]], place)
  y_d = fluid.create_lod_tensor(np.array([[1.1],[1.1],[1.1],[1.1],[1.1],[1.1]]).astype('float32'), [[1,3], [2,1,2,1]],place)


**Execute Computing**

For tensor whose LoD > 1 in Fluid, like data of other types, the order of transfering data is defined by :code:`feed` . In addition, parameter :code:`return_numpy=False` needs to be added to exe.run() to get the output of LoD-Tensor because results are Tensors with LoD information.

.. code-block:: python

  results = exe.run(fluid.default_main_program(),
                    feed={'x':x_d, 'y': y_d },
                    fetch_list=[out],return_numpy=False)

**Check the result of LodTensor**

Because of the special attributes of LoDTensor, you could not print to check the content. The usual solution to the problem is to fetch the LoDTensor as the output of network and then execute  numpy.array(lod_tensor) to transfer LoDTensor into numpy array: 

.. code-block:: python

  np.array(results[0])

Output:

.. code-block:: text

  array([[1.1],[2.2],[3.3],[4.4],[2.2],[3.3],[4.4],[2.2],[3.3],[4.4]])

**Check the length of sequence**

You can get the recursive sequence length of LoDTensor by checking the sequence length:

.. code-block:: python

    results[0].recursive_sequence_lengths()
    
Output

.. code-block:: text
    
    [[1L, 3L, 3L, 3L]]

**Complete Code**

You can check the output by executing the following complete code:

.. code-block:: python
    
    #Load 
    import paddle
    import paddle.fluid as fluid
    import numpy as np
    #Define forward computation
    x = fluid.layers.data(name='x', shape=[1], dtype='float32', lod_level=1)
    y = fluid.layers.data(name='y', shape=[1], dtype='float32', lod_level=2)
    out = fluid.layers.sequence_expand(x=x, y=y, ref_level=0)
    #Define place for computation
    place = fluid.CPUPlace()
    #Create executer
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    #Create LoDTensor
    x_d = fluid.create_lod_tensor(np.array([[1.1], [2.2],[3.3],[4.4]]).astype('float32'), [[1,3]], place)
    y_d = fluid.create_lod_tensor(np.array([[1.1],[1.1],[1.1],[1.1],[1.1],[1.1]]).astype('float32'), [[1,3], [1,2,1,2]], place)
    #Start computing
    results = exe.run(fluid.default_main_program(),
                      feed={'x':x_d, 'y': y_d },
                      fetch_list=[out],return_numpy=False)
    #Output result
    print("The data of the result: {}.".format(np.array(results[0])))
    #print the length of sequence of result
    print("The recursive sequence lengths of the result: {}.".format(results[0].recursive_sequence_lengths()))
    #print the LoD of result
    print("The LoD of the result: {}.".format(results[0].lod()))


Summary
========

Then, we believe that you have known about the concept LoD-Tensor. And an attempt to change x_d and y_d in code above and then to check the output may help you get a better understanding of this flexible structure.

About more model applications of LoDTensor, you can refer to `Word2vec <../../../beginners_guide/basics/word2vec/index_en.html>`_ , `Individual Recommendation <../../../beginners_guide/basics/recommender_system/index_en.html>`_ , `Sentiment Analysis <../../../beginners_guide/basics/understand_sentiment/index_en.html>`_ in the beginner's guide. 

About more difffiult and complex examples of application, please refer to associated information about `models <../../../user_guides/models/index_en.html>`_ .
