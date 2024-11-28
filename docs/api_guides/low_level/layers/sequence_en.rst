.. _api_guide_sequence_en:

########
Sequence
########

Many problems in the field of deep learning involve the processing of the `sequence <https://en.wikipedia.org/wiki/Sequence>`_.
From Wiki's definition, sequences can represent a variety of physical meanings, but in deep learning, the most common is still "time sequence" - a sequence containing information of multiple time steps.

In Paddle Fluid, we represent the sequence as ``DenseTensor``.
Because the general neural network performs computing batch by batch, we use a DenseTensor to store a mini batch of sequences.
The 0th dimension of a DenseTensor contains all the time steps of all sequences in the mini batch, and LoD is used to record the length of each sequence to distinguish different sequences.
In the calculation, it is also necessary to split the 0th dimension of a mini batch in the DenseTensor into a number of sequences according to the LoD information. (Please refer to the LoD related documents for details. )
Therefore, the operation for the 0th dimension of DenseTensor cannot be performed simply by a general layer. The operation of this dimension must be combined with the information of LoD.
(For example, you can't reshape the 0th dimension of a sequence with :code:`layers.reshape`).

In order to correctly implement various sequence-oriented operations, we have designed a series of sequence-related APIs.
In practice, because a DenseTensor contains a mini batch of sequences, and different sequences in the same mini batch usually belong to multiple samples, they do not and should not interact with each other.
Therefore, if a layer is input with two (or more) DenseTensors (or with a list of DenseTensors), and each DenseTensor represents a mini batch of sequences, the first sequence in the first DenseTensor will be only calculated with the first sequence in the second DenseTensor, and the second sequence in the first DenseTensor will only be calculated with the second sequence in the second DenseTensor. To conclude with, the *i'th* sequence in the first DenseTensor will only be calculated with the *i'th* sequence in the second DenseTensor, and so on.

**In summary, a DenseTensor stores multiple sequences in a mini batch, where the number of sequences is batch size; when multiple DenseTensors are calculated, the i'th sequence in each DenseTensor will only be calculated with the i'th of the other DenseTensors. Understanding this is critical to understand the following associated operations.**

1. sequence_softmax
-------------------
This layer takes a mini batch of sequences as input and does a softmax operation in each sequence. The output is a mini batch of sequences in the same shape, but it is normalized by softmax within the sequence.
This layer is often used to do softmax normalization within each sequence.

 Please refer to :ref:`api_fluid_layers_sequence_softmax`


2. sequence_concat
------------------
The layer takes a list as input, which can contain multiple DenseTensors, and every DenseTensors is a mini batch of sequences.
The layer will concatenate the i-th sequence in each batch into a new sequence in the time dimension as the i'th sequence in the returned batch.
Of course, the sequences of each DenseTensor in the list must have the same batch size.

 Please refer to :ref:`api_fluid_layers_sequence_concat`


3. sequence_first_step
----------------------
This layer takes a DenseTensor as input and takes the first element in each sequence (the element of the first time step) as the return value.

 Please refer to :ref:`api_fluid_layers_sequence_first_step`


4. sequence_last_step
---------------------
Same as :code:`sequence_first_step` except that this layer takes the last element in each sequence (i.e. the last time step) as the return value.

 Please refer to :ref:`api_fluid_layers_sequence_last_step`


5. sequence_expand
------------------
This layer has two DenseTensors of sequences as input and extends the sequence in the first batch according to the LoD information of the sequence in the second DenseTensor.
It is usually used to extend a sequence with only one time step (for example, the return result of :code:`sequence_first_step`) into a sequence with multiple time steps, which is convenient for calculations with sequences composed of multiple time steps.

 Please refer to :ref:`api_fluid_layers_sequence_expand`


6. sequence_expand_as
---------------------
This layer takes two DenseTensors of sequences as input and then extends each sequence in the first Tensor to a sequence with the same length as the corresponding one in the second Tensor.
Unlike :code:`sequence_expand` , this layer will strictly extend the sequence in the first DenseTensor to have the same length as the corresponding one in the second DenseTensor.
If it cannot be extended to the same length (for example, the sequence length of the second batch is not an integer multiple of the sequence length of the first batch), an error will be reported.

 Please refer to :ref:`api_fluid_layers_sequence_expand_as`


7. sequence_enumerate
---------------------
This layer takes a DenseTensor of sequences as input and also specifies the length of a :code:`win_size`. This layer will take a subsequence of length :code:`win_size` in all sequences and combine them into a new sequence.

 Please refer to :ref:`api_fluid_layers_sequence_enumerate`


8. sequence_reshape
-------------------
This layer requires a DenseTensor of sequences as input, and you need to specify a :code:`new_dim` as the dimension of the new sequence.
The layer will reshape each sequence in the mini batch to the dimension given by new_dim. Note that the length of each sequence will be changed (so does the LoD information) to accommodate the new shape.

 Please refer to :ref:`api_fluid_layers_sequence_reshape`


9. sequence_scatter
-------------------
This layer can scatter a sequence of data onto another tensor. This layer has three inputs, one is a target tensor to be scattered :code:`input`;
One is the sequence of data to scatter :code:`update` ; One is the upper coordinate of the target tensor :code:`index` . Output is the tensor after scatter, whose shape is the same as :code:`input`.

 Please refer to :ref:`api_fluid_layers_sequence_scatter`


10. sequence_pad
----------------
This layer can pad sequences of unequal length into equal-length sequences. To use this layer you need to provide a :code:`PadValue` and a :code:`padded_length`.
The former is the element used to pad the sequence, it can be a number or a tensor; the latter is the target length of the sequence.
This layer will return the padded sequence, and a tensor :code:`Length` of the length for each sequence before padding.

 Please refer to :ref:`api_fluid_layers_sequence_pad`


11. sequence_mask
-----------------
This layer will generate a mask based on :code:`input`, where the :code:`input` is a tensor that records the length of each sequence.
In addition, this layer requires a parameter :code:`maxlen` to specify the largest sequence length in the sequence.
Usually, this layer is used to generate a mask that will filter away the portion of the paddings in the sequence.
The :code:`input` tensor can usually directly use the returned :code:`Length` from :code:`sequence_pad`  .

 Please refer to :ref:`api_fluid_layers_sequence_mask`
