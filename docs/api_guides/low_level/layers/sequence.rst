..  _api_guide_sequence:

########
序列
########

在深度学习领域许多问题涉及到对 `序列（sequence） <https://en.wikipedia.org/wiki/Sequence>`_ 的处理。
从 Wiki 上的释义可知，序列可以表征多种物理意义，但在深度学习中，最常见的仍然是"时间序列"——一个序列包含多个时间步的信息。

在 Paddle Fluid 中，我们将序列表示为 ``LoDTensor``。
因为一般进行神经网络计算时都是一个 batch 一个 batch 地计算，所以我们用一个 LoDTensor 来存储一个 mini batch 的序列。
一个 LoDTensor 的第 0 维包含该 mini batch 中所有序列的所有时间步，并且用 LoD 来记录各个序列的长度，区分不同序列。
而在运算时，还需要根据 LoD 信息将 LoDTensor 中一个 mini batch 的第 0 维拆开成多个序列。（具体请参考上述 LoD 相关的文档。）
所以，对这类 LoDTensor 第 0 维的操作不能简单地使用一般的 layer 来进行，针对这一维的操作必须要结合 LoD 的信息。
(例如，你不能用 :code:`layers.reshape` 来对一个序列的第 0 维进行 reshape)。

为了实行各类针对序列的操作，我们设计了一系列序列相关的 API，专门用于正确处理序列相关的操作。
实践中，由于一个 LoDTensor 包括一个 mini batch 的序列，同一个 mini batch 中不同的序列通常属于多个 sample，它们彼此之间不会也不应该发生相互作用。
因此，若一个 layer 以两个（或多个）LoDTensor 为输入（或者以一个 list 的 LoDTensor 为输入），每一个 LoDTensor 代表一个 mini batch 的序列，则第一个 LoDTensor 中的第一个序列只会和第二个 LoDTensor 中的第一个序列发生计算，
第一个 LoDTensor 中的第二个序列只会和第二个 LoDTensor 中的第二个序列发生计算，第一个 LoDTensor 中的第 i 个序列只会和第二个 LoDTensor 中第 i 个序列发生计算，依此类推。

**总而言之，一个 LoDTensor 存储一个 mini batch 的多个序列，其中的序列个数为 batch size；多个 LoDTensor 间发生计算时，每个 LoDTensor 中的第 i 个序列只会和其他 LoDTensor 中第 i 个序列发生计算。理解这一点对于理解接下来序列相关的操作会至关重要。**

1. sequence_softmax
-------------------
这个 layer 以一个 mini batch 的序列为输入，在每个序列内做 softmax 操作。其输出为一个 mini batch 相同 shape 的序列，但在序列内是经 softmax 归一化过的。
这个 layer 往往用于在每个 sequence 内做 softmax 归一化。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_softmax`


2. sequence_concat
------------------
这个 layer 以一个 list 为输入，该 list 中可以含有多个 LoDTensor，每个 LoDTensor 为一个 mini batch 的序列。
该 layer 会将每个 batch 中第 i 个序列在时间维度上拼接成一个新序列，作为返回的 batch 中的第 i 个序列。
理所当然地，list 中每个 LoDTensor 的序列必须有相同的 batch size。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_concat`


3. sequence_first_step
----------------------
这个 layer 以一个 LoDTensor 作为输入，会取出每个序列中的第一个元素（即第一个时间步的元素），并作为返回值。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_first_step`


4. sequence_last_step
---------------------
同 :code:`sequence_first_step` ，除了本 layer 是取每个序列中最后一个元素（即最后一个时间步）作为返回值。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_last_step`


5. sequence_expand
------------------
这个 layer 有两个 LoDTensor 的序列作为输入，并按照第二个 LoDTensor 中序列的 LoD 信息来扩展第一个 batch 中的序列。
通常用来将只有一个时间步的序列（例如 :code:`sequence_first_step` 的返回结果）延展成有多个时间步的序列，以此方便与有多个时间步的序列进行运算。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_expand`


6. sequence_expand_as
---------------------
这个 layer 需要两个 LoDTensor 的序列作为输入，然后将第一个 Tensor 序列中的每一个序列延展成和第二个 Tensor 中对应序列等长的序列。
不同于 :code:`sequence_expand` ，这个 layer 会将第一个 LoDTensor 中的序列严格延展为和第二个 LoDTensor 中的序列等长。
如果无法延展成等长的（例如第二个 batch 中的序列长度不是第一个 batch 中序列长度的整数倍），则会报错。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_expand_as`


7. sequence_enumerate
---------------------
这个 layer 需要一个 LoDTensor 的序列作为输入，同时需要指定一个 :code:`win_size` 的长度。这个 layer 将依次取所有序列中长度为 :code:`win_size` 的子序列，并组合成新的序列。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_enumerate`


8. sequence_reshape
-------------------
这个 layer 需要一个 LoDTensor 的序列作为输入，同时需要指定一个 :code:`new_dim` 作为新的序列的维度。
该 layer 会将 mini batch 内每个序列 reshape 为 new_dim 给定的维度。注意，每个序列的长度会改变（因此 LoD 信息也会变），以适应新的形状。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_reshape`


9. sequence_scatter
-------------------
这个 layer 可以将一个序列的数据 scatter 到另一个 tensor 上。这个 layer 有三个 input，一个要被 scatter 的目标 tensor :code:`input`；
一个是序列的数据 :code:`update` ，一个是目标 tensor 的上坐标 :code:`index` 。Output 为 scatter 后的 tensor，形状和 :code:`input` 相同。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_scatter`


10. sequence_pad
----------------
这个 layer 可以将不等长的序列补齐成等长序列。使用这个 layer 需要提供一个 :code:`PadValue` 和一个 :code:`padded_length`。
前者是用来补齐序列的元素，可以是一个数也可以是一个 tensor；后者是序列补齐的目标长度。
这个 layer 会返回补齐后的序列，以及一个记录补齐前各个序列长度的 tensor :code:`Length`。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_pad`


11. sequence_mask
-----------------
这个 layer 会根据 :code:`input` 生成一个 mask，:code:`input` 是一个记录了每个序列长度的 tensor。
此外这个 layer 还需要一个参数 :code:`maxlen` 用于指定序列中最长的序列长度。
通常这个 layer 用于生成一个 mask，将被 pad 后的序列中 pad 的部分过滤掉。
:code:`input` 的长度 tensor 通常可以直接用 :code:`sequence_pad` 返回的 :code:`Length`。

API Reference 请参考 :ref:`cn_api_fluid_layers_sequence_mask`
