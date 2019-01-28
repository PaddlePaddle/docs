.. _api_guide_sequence:

########
Sequence
########

Many problems in the field of deep learning involve the processing of the `sequence <https://en.wikipedia.org/wiki/Sequence>`_.
From the  wiki's definition, sequences can be characterized by a variety of physical meanings, but in deep learning, the most common is still "time sequence" - a sequence of information containing  multiple time steps.

In Paddle Fluid, we represent the sequence as :ref:`cn_api_fluid_LoDTensor`.
Because the general neural network calculation is a batch and a batch calculation, we use a LoDTensor to store a sequence of mini batch.
The 0th dimension of a LoDTensor contains all the time steps of all sequences in the mini batch, and LoD is used to record the length of each sequence to distinguish different sequences.
In the calculation, it is also necessary to split the 0th dimension of a mini batch in the LoDTensor into a plurality of sequences according to the LoD information. (Please refer to the LoD related documents above for details.)
Therefore, the operation of the 0th dimension to this kind of LoDTensor cannot be performed simply by using the general layer. The operation of this dimension must be combined with the information of LoD.
(For example, you can't reshape the 0th dimension of a sequence with :code:`layers.reshape`).

In order to implement various sequence-oriented operations, we have designed a series of sequence-related APIs designed to correctly handle sequence-related operations.
In practice, since a LoDTensor includes a sequence of mini batch, different sequences in the same mini batch usually belong to multiple samples, and they do not and should not interact with each other.
Therefore, if a layer is input with two (or more) LoDTensors (or with a list of LoDTensors), each LoDTensor represents a sequence of mini batch, then the first sequence in the first LoDTensor will  be only calculate with the first sequence in the second LoDTensor, and the second sequence in the first LoDTensor will only be calculated with the second sequence in the second LoDTensor. The ith sequence in the first LoDTensor will only be calculated with the ith sequence in the second LoDTensor, and so on.

**In summary, a LoDTensor stores multiple sequences of a mini batch, where the number of sequences is batch size; when multiple LoDTensors are calculated, the ith sequence in each LoDTensor will only be calculated with the ith of the other LoDTensors. Understanding this is critical to understanding the operations associated with the next sequence.**

1. Sequence_softmax
-------------------
This layer takes a sequence of mini batch as input and does a softmax operation in each sequence. The output is a sequence of the same shape of the mini batch, but it is normalized by softmax within the sequence.
This layer is often used to do softmax normalization within each sequence.

API Reference Please refer to :ref:`cn_api_fluid_layers_sequence_softmax`


2. sequence_concat
------------------
The layer takes a list as input, which can contain multiple LoDTensors, every LoDTensors is a sequence of mini batches.
The layer will splicing the i-th sequence in each batch into a new sequence in the time dimension as the ith sequence in the returned batch.
Of course, the sequence of each LoDTensor in the list must have the same batch size.

API Reference Please refer to :ref:`cn_api_fluid_layers_sequence_concat`


3. sequence_first_step
----------------------
This layer takes a LoDTensor as input and takes the first element in each sequence (the element of the first time step) as a return value.

API Reference please refer to :ref:`cn_api_fluid_layers_sequence_first_step`


4. sequence_last_step
---------------------
Same as :code:`sequence_first_step` except that the layer takes the last element in each sequence (ie the last time step) as the return value.

API Reference please refer to :ref:`cn_api_fluid_layers_sequence_last_step`


5. sequence_expand
------------------
This layer has two sequences of LoDTensor as input and extends the sequence in the first batch according to the LoD information of the sequence in the second LoDTensor.
It is usually used to extend a sequence with only one time step (for example, the return result of :code:`sequence_first_step`) into a sequence with multiple time steps, which is convenient for operations with multiple time steps's sequence.

API Reference please refer to :ref:`cn_api_fluid_layers_sequence_expand`


6. sequence_expand_as
---------------------
This layer takes the sequence of two LoDTensors as input and then extends each sequence in the first Tensor sequence into a sequence of equal length to the corresponding sequence in the second Tensor.
Unlike :code:`sequence_expand` , this layer will strictly extend the sequence in the first LoDTensor to be the same length as the sequence in the second LoDTensor.
If it cannot be extended to the same length (for example, the sequence length of the second batch is not an integer multiple of the sequence length of the first batch), an error will be reported.

API Reference please refer to :ref:`cn_api_fluid_layers_sequence_expand_as`


7. sequence_enumerate
---------------------
This layer takes a sequence of LoDTensor as input and also specifies the length of a :code:`win_size`. This layer will take a subsequence of length :code:`win_size` in all sequences and combine them into a new sequence.

API Reference Please refer to :ref:`cn_api_fluid_layers_sequence_enumerate`


8. sequence_reshape
-------------------
This layer requires a sequence of LoDTensor as input, and you need to specify a :code:`new_dim` as the dimension of the new sequence.
The layer will reshape each sequence of the mini batch to the dimension given by new_dim. Note that the length of each sequence will be changed (so does the LoD information) to accommodate the new shape.

API Reference See :ref:`cn_api_fluid_layers_sequence_reshape`


9. sequence_scatter
-------------------
This layer can scatter a sequence of data onto another tensor. This layer has three inputs, one is a target tensor to be scattered :code:`input`;
One is the sequence of data :code:`update` , one is the upper coordinate of the target tensor :code:`index` . Output is the tensor after scatter, which the shape is the same as :code:`input`.

API Reference See :ref:`cn_api_fluid_layers_sequence_scatter`


10. sequence_pad
----------------
This layer can compile sequences of unequal length into equal length sequences. To use this layer you need to provide a :code:`PadValue` and a :code:`padded_length`.
The former is the element used to complete the sequence, it can be a number or a tensor; the latter is the target length of the sequence.
This layer will return the completed sequence, and a tensor :code:`Length` of the length for each sequence before the record is completed.

API Reference Please refer to :ref:`cn_api_fluid_layers_sequence_pad`


11. sequence_mask
-----------------
This layer will generate a mask based on :code:`input`, where the :code:`input` is a tensor that records the length of each sequence.
In addition, this layer requires a parameter :code:`maxlen` to specify the longest sequence length in the sequence.
Usually this layer is used to generate a mask that will be filtered out by the portion of the pad in the sequence.
The length of :code:`input` tensor can usually directly use the :code:`sequence_pad` 's  returned :code:`Length`.

API Reference please refer to :ref:`cn_api_fluid_layers_sequence_mask`
