.. _api_guide_tensor_en:

########
Tensor
########

There are two data structures used in Fluid to host the data, namely `Tensor and LoD_Tensor <../../../user_guides/howto/basic_concept/lod_tensor_en.html>`_ .  LoD-Tensor is a unique concept of Fluid, which appends sequence information to Tensor. The data that can be transferred in the framework includes: input, output, and learnable parameters in the network. All of them are uniformly represented by LoD-Tensor. In addition, tensor can be regarded as a special LoD-Tensor.

Now let's take a closer look at the operations related to these two types of data.

Tensor
======

1. create_tensor
---------------------
Tensor is used to carry data in the framework, using :code:`create_tensor` to create a Lod-Tensor variable of the specified the data type.

API reference : :ref:`api_fluid_layers_create_tensor`


2. create_parameter
---------------------
The neural network training process is a learning process for parameters. Fluid uses :code:`create_parameter` to create a learnable parameter. The value of this parameter can be changed by the operator.

API reference  : :ref:`api_fluid_layers_create_parameter`



3. create_global_var
---------------------
Fluid uses :code:`create_global_var` to create a global tensor and this API allows you to specify the data type, shape, and value of the Tensor variable being created.

API reference  : :ref:`api_fluid_layers_create_global_var`


4. cast
---------------

Fluid uses :code:`cast` to convert the data to the specified type.

API reference  : :ref:`api_fluid_layers_cast`


5.concat
----------------

Fluid uses :code:`concat` to concatenate input data along a specified dimension.

API reference  : :ref:`api_fluid_layers_concat`


6. sums
----------------

Fluid uses :code:`sums` to sum up the input data.

API reference  : :ref:`api_fluid_layers_sums`

7. fill_constant
-----------------

Fluid uses :code:`fill_constant` to create a Tensor with a specific shape and type. The initial value of this variable can be set via :code:`value`.

API reference : :ref:`api_fluid_layers_fill_constant`

8. assign
---------------

Fluid uses :code:`assign` to duplicate a variable.

API reference  : :ref:`api_fluid_layers_assign`

9. argmin
--------------

Fluid uses :code:`argmin` to calculate the index of the smallest element on the specified axis of Tensor.

API reference  : :ref:`api_fluid_layers_argmin`

10. argmax
-----------

Fluid uses :code:`argmax` to calculate the index of the largest element on the specified axis of Tensor.

API reference  : :ref:`api_fluid_layers_argmax`

11. argsort
------------

Fluid uses :code:`argsort` to sort the input Tensor on the specified axis and it will return the sorted data variables and their corresponding index values.

API reference : :ref:`api_fluid_layers_argsort`

12. ones
-------------

Fluid uses :code:`ones` to create a Tensor of the specified size and data type with an initial value of 1.

API reference : :ref:`api_fluid_layers_ones`

13. zeros
---------------

Fluid uses :code:`zeros` to create a Tensor of the specified size and data type with an initial value of zero.

API reference : :ref:`api_fluid_layers_zeros`

14. reverse
-------------------

Fluid uses :code:`reverse` to invert Tensor along the specified axis.

API reference : :ref:`api_fluid_layers_reverse`



LoD-Tensor
============

LoD-Tensor is very suitable for sequence data. For related knowledge, please read `Tensor and LoD_Tensor <../../../user_guides/howto/basic_concept/lod_tensor_en.html>`_ .

1.create_lod_tensor
-----------------------

Fluid uses :code:`create_lod_tensor` to create a LoD_Tensor with new hierarchical information based on a numpy array, a list, or an existing LoD_Tensor.

API reference : :ref:`api_fluid_create_lod_tensor`

2. create_random_int_lodtensor
----------------------------------

Fluid uses :code:`create_random_int_lodtensor` to create a LoD_Tensor composed of random integers.

API reference : :ref:`api_fluid_create_random_int_lodtensor`

3. reorder_lod_tensor_by_rank
---------------------------------

Fluid uses :code:`reorder_lod_tensor_by_rank` to reorder the sequence information of the input LoD_Tensor in the specified order.

API reference : :ref:`api_fluid_layers_reorder_lod_tensor_by_rank`
