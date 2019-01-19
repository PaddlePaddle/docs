##################
LoD-Tensor User Guide
##################

LoD(Level-of-Detail) Tensor is a typical terminology in Fluid with sequence information appended to Tensor. Data transferred in Fluid contains leanable parameters of input,output and network,all of which are represented by LoD-Tensor.

With help of the user guide,you will learn about the design philosophy of LoD-Tensor in Fluid so that you can use such a data type more flexiblely.

Challenge of variable-length sequence
================

In most deep learning frameworks, a mini-batch is represented by Tensor.

For example, if there are 10 pictures in a mini-batch and the size of each picture is 32*32, the mini-batch is a 10*32*32 Tensor.

Or in the NLP task, there are N sentences in a mini-batch and the length of each sentence is L. Every word is represented by a one-hot vector with D dimensions.Then the mini-batch can be represented by N*L*D Tensor.

In the two examples above,the size of sequence element is united. However, train data is variable-length sequence under many circumstances. In this situatin, method to be taken in most frameworks is to set a fixed length and those shorter sequence data will be padded with 0 to the fixed length.

Owing to the LoD-Tensor in Fluid, it is not necessary to keep the length of sequence data in every mini-batch united.Therefore tasks sensible to sequence like NLP can also be finished without padding.

Index Data Structure (LoD) is induced to Fluid to split Tensor into sequence.

LoD Index
===========

To have a better understanding of the concept of LoD, you can refer to cases in the section.

** mini-batch consisting of sentences**

Supposing a mini-batch contains three sentences,ever of which contains 3, 1, 2 words respectively. Then the mini-batch can be represented by (3+1+2)*D Tensor with some index information:

.. code-block :: text

  3       1   2
  | | |   |   | |

In the text above, each :code:`|` represents a word vector with D dimension and a 1-level LoD is made up of number 3,1,2.

**recursive sequence**

Take a 2-level LoD-Tensor for example,a mini-batch contains articles of 3 sentences,1 sentences and 2 sentences. The number of words in every sentence is different.Then the mini-batch is formed as follows:

.. code-block:: text


  3            1 2
  3   2  4     1 2  3
  ||| || ||||  | || |||


Represented LoD:

.. code-block:: text

  [[3，1，2]/*level=0*/，[3，2，4，1，2，3]/*level=1*/]


**mini-batch in video**

In the task of vision,it usually needs to deal objects with high dimension like vedios and pistures.Supposing a mini-batch contains 3 videos, which is composed of 3 frames, 1 frames, 2 frames respectively.The size of each frame is 640*480.Then the mini-batch can be described as:

.. code-block:: text

  3     1  2
  口口口 口 口口


The size of the tensor at the bottom is (3+1+2)*640*480.Every :code:`口` represents a 640*480 picture.

**mini-batch in picture**

Traditionally,for a mini-batch of N pictures with fixed size, LoD-Tensor is described as:

.. code-block:: text

  1 1 1 1     1
  口口口口 ... 口

Under such circumstance, we will consider LoD-Tensor as a common tensor instead of ingoring information because of the index value 1.

.. code-block:: text

  口口口口 ... 口

**model parameter**

model parameter is a common tensor which is described as a 0-level LoD-Tensor in Fluid.

LoDTensor expressed by offset
=====================

To have a quick visit to fundamental sequnce,you can take the offset expression method——store the first and last element of sequence instead of length.

In the example above, you can compute the length of fundamental elements:

.. code-block:: text

  3 2 4 1 2 3

It is expressed by offset as follows:

.. code-block:: text

  0  3  5   9   10  12   15
     =  =   =   =   =    =
     3  2+3 4+5 1+9 2+10 3+12

Therefore we infer that the first sentence starts from word 0 to word 3 and the second sentence starts from word 3 to word 5.

Similarly,for the length of the top layer of LoD

.. code-block:: text

  3 1 2

It can be expressed by offset:

.. code-block:: text

  0 3 4   6
    = =   =
    3 3+1 4+2

Therefore the LoD-Tensor is expressed by offset:：

.. code-block:: text

  0       3    4      6
    3 5 9   10   12 15


LoD-Tensor
=============
A LoD-Tensor can be regarded as a tree in which the leaf is a symbol of fundamental sequence element and branch is a symbol of flag of fundamental element.

There are two ways to express sequence information of LoD-Tensor in Fluid: primary length and offset. LoD-Tensor is expressed by offset in Paddle to offer a quicker visit to sequence;LoD-Tensor is expressed by primary length in python API to make user understand and compute more easily.The primary length is named as  :code:`recursive_sequence_lengths` 。

Take a 2-level LoD-Tensor mentioned above as an example:

.. code-block:: text

  3           1  2
  3   2  4    1  2  3
  ||| || |||| |  || |||

- LoD-Tensor expressed by offset: [ [0,3,4,6] , [0,3,5,9,10,12,15] ]，
- LoD-Tensor expressed by primary length: recursive_sequence_lengths=[ [3-0 , 4-3 , 6-4] , [3-0 , 5-3 , 9-5 , 10-9 , 12-10 , 15-12] ]。


Take text sequence as an example,[3,1,2] indicates 3 articles in the mini-batch,which contains 3,1,2 sentences respectively.[3,2,4,1,2,3] indicates 3,2,4,1,2,3 words in sentences respectively.

recursive_seq_lens is Double Layer nested list,saying the element of the list is list.The size of the outermost list represents the nested layers,saying the size of lod-level;Each inner list represents the size of each element in each lod-level. 

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
  print len(a.recursive_sequence_lengths())
  # output：2

  #Check the number of the most fundamental elements
  print sum(a.recursive_sequence_lengths()[-1])
  # output:15 (3+2+4+1+2+3=15)

Code examples
===========

Input variable x is expanded according to specified layer level y-lod in the code in this section.The example below contains some fundamental conceptions of LoD-Tensor.By following the code,you will

-  Have a direct understanding of the implementation of :code:`fluid.layers.sequence_expand` in Fluid
-  Know how to create LoD-Tensor in Fluid
-  Learn how to print the content of LoDTensor


  
**Define the Process of Computing**

layers.sequence_expand expands x by obtaining the lod value of y. About more explanation of :code:`fluid.layers.sequence_expand` ,please read :ref:`api_fluid_layers_sequence_expand` first. 

Code of expanding sequence:

.. code-block:: python

  x = fluid.layers.data(name='x', shape=[1], dtype='float32', lod_level=0)
  y = fluid.layers.data(name='y', shape=[1], dtype='float32', lod_level=1)
  out = fluid.layers.sequence_expand(x=x, y=y, ref_level=0)

*Note*：The dimension of ouput LoD-Tensor is only associated with the dimension of real data transferred in.The shape value set for x and y in the definition of network structure is just as a placeholder with little influence on the result.  

**Create Executor**

.. code-block:: python

  place = fluid.CPUPlace()
  exe = fluid.Executor(place)
  exe.run(fluid.default_startup_program())

**Prepare Data**

Here we use :code:`fluid.create_lod_tensor` to create the input data of :code:`sequence_expand` and expand x_d by defining LoD of y_d. The output value is only associated with LoD of y_d. And the data of y_d is not invovled in the process of computation.The dimension can keep same with LoD[-1]

About the user guide of :code:`fluid.create_lod_tensor()`, please refer to :ref:`api_fluid_create_lod_tensor` 。

Code：

.. code-block:: python

  x_d = fluid.create_lod_tensor(np.array([[1.1],[2.2],[3.3],[4.4]]).astype('float32'), [[1,3]], place)
  y_d = fluid.create_lod_tensor(np.array([[1.1],[1.1],[1.1],[1.1],[1.1],[1.1]]).astype('float32'), [[1,3], [2,1,2,1]],place)


**Execute Computing**

For tensor whose LoD > 1 in Fluid like data of other types,the order of transfering data is defined by :code:`feed` . In addition,parameter :code:`return_numpy=False` needs to be added to exe.run() to get the output of LoD-Tensor because results is a Tensor with LoD information.

.. code-block:: python

  results = exe.run(fluid.default_main_program(),
                    feed={'x':x_d, 'y': y_d },
                    fetch_list=[out],return_numpy=False)

**Check the result of LodTensor**

Because of the special attributes of LoDTensor,you could not print to check the content.The usual solution to solve the problem is to fetch the LoDTensor as the output of network and then execute  numpy.array(lod_tensor) to transfer LoDTensor into numpy array: 

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
    x = fluid.layers.data(name='x', shape=[1], dtype='float32', lod_level=0)
    y = fluid.layers.data(name='y', shape=[1], dtype='float32', lod_level=1)
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

Then,we believe that you have known about the concept LoD-Tensor.And an attempt to change x_d and y_d in code above and then to check the output may help you get a better understanding of the flexible structure.

About more applications of LoDTensor model,you can refer to `Word2vec <../../../beginners_guide/basics/word2vec/index.html>`_ 、`Indivisual Recommandation <../../../beginners_guide/basics/recommender_system/index.html>`_、`Sentimental Analysis <../../../beginners_guide/basics/understand_sentiment/index.html>`_ in the beginner's guide. 

About more difffiult and complex examples of application,please refer to associated information about `models <../../../user_guides/models/index_cn.html>`_ 