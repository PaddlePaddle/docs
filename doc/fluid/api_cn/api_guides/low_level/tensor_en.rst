.. _api_guide_tensor_en:

########
Tensor
########

Two data structures are used in Fluid to host the data, namely `Tensor and LoD_Tensor <../../../../user_guides/howto/prepare_data/lod_tensor.html>`_. Among them, LoD-Tensor is a unique concept of Fluid, which adds sequence information to Tensor. The data that can be transmitted in the framework includes: input, output, and learnable parameters in the network. All of them are uniformly represented by LoD-Tensor. Tensor can be regarded as a special LoD-Tensor.

The following describes the operations related to these two types of data.

Tensor
=======

Create_tensor
---------------------
Tensor is used to carry data in the framework, using :code:`create_tensor` to create a Lod-Tensor variable that specifies the data type.

API reference See: :ref:`api_fluid_layers_create_tensor`


2. create_parameter
---------------------
The neural network training process is a learning process for parameters. Fluid uses :code:`create_parameter` to create a learnable parameter. The value of this parameter can be changed by the operator.

API reference Please refer to ::ref:`api_fluid_layers_create_parameter`



3. create_global_var
---------------------
Fluid uses :code:`create_global_var` to create a global tensor that allows you to specify the data type, shape, and value of the Tensor variable being created.

API reference Please refer to ::ref:`api_fluid_layers_create_global_var`


4. cast
---------------

Fluid uses :code:`cast` to convert the data to the specified type.

API reference Please refer to ::ref:`api_fluid_layers_cast`


Concat
----------------

Fluid uses :code:`concat` to connect input data along a specified dimension.

API reference Please refer to ::ref:`api_fluid_layers_concat`


6. sums
----------------

Fluid uses :code:`sums` to perform an addition to the input data.

API reference Please refer to ::ref:`api_fluid_layers_sums`


7. fill_constant_batch_size_like
---------------------------------

Fluid uses :code:`fill_constant_batch_size_like` to create a Tensor with a specific shape, type, and batch_size. And the initial value of the Tensor can be specified as an arbitrary constant. The batch_size information is determined by the tensor's :code:`input_dim_idx` and :code:`output_dim_idx`.

API reference Please refer to ::ref:`api_fluid_layers_fill_constant_batch_size_like`

8. fill_constant
-----------------

Fluid uses :code:`fill_constant` to create a Tensor with a specific shape and type. The initial value of this variable can be set via :code:`value`.

API reference See: :ref:`api_fluid_layers_fill_constant`

9. assign
---------------

Fluid uses :code:`assign` to copy a variable.

API reference Please refer to ::ref:`api_fluid_layers_assign`

10. argmin
--------------

Fluid uses :code:`argmin` to calculate the index of the smallest element on the specified axis of Tensor.

API reference Please refer to ::ref:`api_fluid_layers_assign`

11. argmax
-----------

Fluid uses :code:`argmax` to calculate the index of the largest element on the specified axis of Tensor.

API reference Please refer to ::ref:`api_fluid_layers_argmax`

12. argsort
------------

Fluid uses :code:`argsort` to sort the input Tensor on the specified axis and return the sorted data variables and their corresponding index values.

API reference See: :ref:`api_fluid_layers_argsort`

13. ones
-------------

Fluid uses :code:`ones` to create a Tensor of the specified size and data type with an initial value of 1.

API reference See: :ref:`api_fluid_layers_ones`

14. zeros
---------------

Fluid uses :code:`zeros` to create a Tensor of the specified size and data type with an initial value of zero.

API reference See: :ref:`api_fluid_layers_zeros`

15. reverse
-------------------

Fluid uses :code:`reverse` to invert Tensor along the specified axis.

API reference See: :ref:`api_fluid_layers_reverse`



LoD-Tensor
============

LoD-Tensor is very suitable for sequence data. For related knowledge, please read `LoD_Tensor <../../../../user_guides/howto/prepare_data/lod_tensor.html>`_.

Create_lod_tensor
-----------------------

Fluid uses :code:`create_lod_tensor` to create a LoD_Tensor with new hierarchical information based on a numpy array, list, or existing LoD_Tensor.

API reference See: :ref:`api_fluid_create_lod_tensor`

2. create_random_int_lodtensor
----------------------------------

Fluid uses :code:`create_random_int_lodtensor` to create a LoD_Tensor of random integers.

API reference See: :ref:`api_fluid_create_random_int_lodtensor`

3. reorder_lod_tensor_by_rank
---------------------------------

Fluid uses :code:`reorder_lod_tensor_by_rank` to retake the sequence information of the input LoD_Tensor in the specified order.

API reference See: ref:`api_fluid_layers_reorder_lod_tensor_by_rank`