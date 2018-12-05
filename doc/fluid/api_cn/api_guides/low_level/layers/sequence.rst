..  _api_guide_sequence:

########
序列
########

在深度学习领域许多问题涉及到对 `序列（sequence） <https://en.wikipedia.org/wiki/Sequence>`_ 的处理。
从Wiki上的释义可知，序列可以表征多种物理意义，但在深度学习中，最常见的仍然是"时间序列"——一个序列包含多个时间步的信息。

在Paddle Fluid中，我们将序列表示为 :ref:`api_fluid_LoDTensor` 。
因为一般进行神经网络计算时都是一个batch一个batch地计算，所以我们用一个LoDTensor来存储一个mini batch的序列。
一个LoDTensor的第0维包含该mini batch中所有序列的所有时间步，并且用LoD来记录各个序列的长度，区分不同序列。
而在运算时，还需要根据LoD信息将LoDTensor中一个mini batch的第0维拆开成多个序列。（具体请参考上述LoD相关的文档。）
所以，对这类LoDTensor第0维的操作不能简单地使用一般的layer来进行，针对这一维的操作必须要结合LoD的信息。
(例如，你不能用 :code:`layers.reshape` 来对一个序列的第0维进行reshape)。

为了实行各类针对序列的操作，我们设计了一系列序列相关的API，专门用于正确处理序列相关的操作。
实践中，由于一个LoDTensor包括一个mini batch的序列，同一个mini batch中不同的序列通常属于多个sample，它们彼此之间不会也不应该发生相互作用。
因此，若一个layer以两个（或多个）LoDTensor为输入（或者以一个list的LoDTensor为输入），每一个LoDTensor代表一个mini batch的序列，则第一个LoDTensor中的第一个序列只会和第二个LoDTensor中的第一个序列发生计算，
第一个LoDTensor中的第二个序列只会和第二个LoDTensor中的第二个序列发生计算，第一个LoDTensor中的第i个序列只会和第二个LoDTensor中第i个序列发生计算，依此类推。

**总而言之，一个LoDTensor存储一个mini batch的多个序列，其中的序列个数为batch size；多个LoDTensor间发生计算时，每个LoDTensor中的第i个序列只会和其他LoDTensor中第i个序列发生计算。理解这一点对于理解接下来序列相关的操作会至关重要。**

1. sequence_softmax
-------------------
这个layer以一个mini batch的序列为输入，在每个序列内做softmax操作。其输出为一个mini batch相同shape的序列，但在序列内是经softmax归一化过的。
这个layer往往用于在每个sequence内做softmax归一化。

API Reference 请参考 :ref:`api_fluid_layers_sequence_softmax`


2. sequence_concat
------------------
这个layer以一个list为输入，该list中可以含有多个LoDTensor，每个LoDTensor为一个mini batch的序列。
该layer会将每个batch中第i个序列在时间维度上拼接成一个新序列，作为返回的batch中的第i个序列。
理所当然地，list中每个LoDTensor的序列必须有相同的batch size。

API Reference 请参考 :ref:`api_fluid_layers_sequence_concat`


3. sequence_first_step
----------------------
这个layer以一个LoDTensor作为输入，会取出每个序列中的第一个元素（即第一个时间步的元素），并作为返回值。

API Reference 请参考 :ref:`api_fluid_layers_sequence_first_step`


4. sequence_last_step
---------------------
同 :code:`sequence_first_step` ，除了本layer是取每个序列中最后一个元素（即最后一个时间步）作为返回值。

API Reference 请参考 :ref:`api_fluid_layers_sequence_last_step`


5. sequence_expand
------------------
这个layer有两个LoDTensor的序列作为输入，并按照第二个LoDTensor中序列的LoD信息来扩展第一个batch中的序列。
通常用来将只有一个时间步的序列（例如 :code:`sequence_first_step` 的返回结果）延展成有多个时间步的序列，以此方便与有多个时间步的序列进行运算。

API Reference 请参考 :ref:`api_fluid_layers_sequence_expand`


6. sequence_expand_as
---------------------
这个layer需要两个LoDTensor的序列作为输入，然后将第一个Tensor序列中的每一个序列延展成和第二个Tensor中对应序列等长的序列。
不同于 :code:`sequence_expand` ，这个layer会将第一个LoDTensor中的序列严格延展为和第二个LoDTensor中的序列等长。
如果无法延展成等长的（例如第二个batch中的序列长度不是第一个batch中序列长度的整数倍），则会报错。

API Reference 请参考 :ref:`api_fluid_layers_sequence_expand_as`


7. sequence_enumerate
---------------------
这个layer需要一个LoDTensor的序列作为输入，同时需要指定一个 :code:`win_size` 的长度。这个layer将依次取所有序列中长度为 :code:`win_size` 的子序列，并组合成新的序列。

API Reference 请参考 :ref:`api_fluid_layers_sequence_enumerate`


8. sequence_reshape
-------------------
这个layer需要一个LoDTensor的序列作为输入，同时需要指定一个 :code:`new_dim` 作为新的序列的维度。
该layer会将mini batch内每个序列reshape为new_dim给定的维度。注意，每个序列的长度会改变（因此LoD信息也会变），以适应新的形状。

API Reference 请参考 :ref:`api_fluid_layers_sequence_reshape`


9. sequence_scatter
-------------------
这个layer可以将一个序列的数据scatter到另一个tensor上。这个layer有三个input，一个要被scatter的目标tensor :code:`input`；
一个是序列的数据 :code:`update` ，一个是目标tensor的上坐标 :code:`index` 。Output为scatter后的tensor，形状和 :code:`input` 相同。

API Reference 请参考 :ref:`api_fluid_layers_sequence_scatter`


10. sequence_pad
----------------
这个layer可以将不等长的序列补齐成等长序列。使用这个layer需要提供一个 :code:`PadValue` 和一个 :code:`padded_length`。
前者是用来补齐序列的元素，可以是一个数也可以是一个tensor；后者是序列补齐的目标长度。
这个layer会返回补齐后的序列，以及一个记录补齐前各个序列长度的tensor :code:`Length`。

API Reference 请参考 :ref:`api_fluid_layers_sequence_pad`


11. sequence_mask
-----------------
这个layer会根据 :code:`input` 生成一个mask，:code:`input` 是一个记录了每个序列长度的tensor。
此外这个layer还需要一个参数 :code:`maxlen` 用于指定序列中最长的序列长度。
通常这个layer用于生成一个mask，将被pad后的序列中pad的部分过滤掉。
:code:`input` 的长度tensor通常可以直接用 :code:`sequence_pad` 返回的 :code:`Length`。

API Reference 请参考 :ref:`api_fluid_layers_sequence_mask`

