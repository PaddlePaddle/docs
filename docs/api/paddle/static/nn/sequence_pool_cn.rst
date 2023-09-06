.. _cn_api_fluid_layers_sequence_pool:

sequence_pool
-------------------------------


.. py:function:: paddle.static.nn.sequence_pool(input, pool_type, is_test=False, pad_value=0.0)



.. note::
该 API 的输入只能是带有 LoD 信息的 Tensor，如果您需要处理的输入是 Tensor 类型，请使用 :ref:`paddle.nn.functional.avg_pool2d <cn_api_nn_functional_avg_pool2d>`  或 :ref:`paddle.nn.functional.max_pool2d <cn_api_nn_functional_max_pool2d>` 。

对输入的 Tensor 进行指定方式的池化（pooling）操作。通过指定 pool_type 参数，将输入的每个序列（sequence）在最后一层 lod_level 上或时间步（time-step）上对特征进行诸如 sum、average、sqrt 等池化操作。

支持六种 pool_type:

- **average**: :math:`Out[i] = \frac{\sum_{i}X_{i}}{N}`
- **sum**: :math:`Out[i] = \sum _{j}X_{ij}`
- **sqrt**: :math:`Out[i] = \frac{ \sum _{j}X_{ij}}{\sqrt{len(\sqrt{X_{i}})}}`
- **max**: :math:`Out[i] = max(X_{i})`
- **last**: :math:`Out[i] = X_{N\_i}`
- **first**: :math:`Out[i] = X_{0}`

其中 ``N_i`` 为待池化第 i 个输入序列的长度。

::

    Case 1:

        input 是 1-level 的 Tensor，且 pad_value = 0.0:
            input.lod = [[0, 2, 5, 7, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]
        输出为 Tensor：
            out.shape = [4, 1]
            其中 out.shape[0] == len(x.lod[-1]) == 4
        对于不同的 pool_type：
            average: out.data = [[2.], [4.], [3.], [0.0]], where 2.=(1. + 3.)/2, 4.=(2. + 4. + 6.)/3, 3.=(5. + 1.)/2
            sum    : out.data = [[4.], [12.], [6.], [0.0]], where 4.=1. + 3., 12.=2. + 4. + 6., 6.=5. + 1.
            sqrt   : out.data = [[2.82], [6.93], [4.24], [0.0]], where 2.82=(1. + 3.)/sqrt(2), 6.93=(2. + 4. + 6.)/sqrt(3), 4.24=(5. + 1.)/sqrt(2)
            max    : out.data = [[3.], [6.], [5.], [0.0]], where 3.=max(1., 3.), 6.=max(2., 4., 6.), 5.=max(5., 1.)
            last   : out.data = [[3.], [6.], [1.], [0.0]], where 3.=last(1., 3.), 6.=last(2., 4., 6.), 1.=last(5., 1.)
            first  : out.data = [[1.], [2.], [5.], [0.0]], where 1.=first(1., 3.), 2.=first(2., 4., 6.), 5.=first(5., 1.)

        上述 out.data 中的最后一个[0.0]均为填充的数据。

    Case 2:

        input 是 2-level 的 Tensor，包含 3 个长度分别为[2, 0, 3]的序列，其中中间的 0 表示序列为空。
        第一个长度为 2 的序列包含 2 个长度分别为[1, 2]的子序列；
        最后一个长度为 3 的序列包含 3 个长度分别为[1, 0, 3]的子序列。
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        以 pool_type 取值为 sum 为例，将根据最后一层的 lod 信息[0, 1, 3, 4, 4, 7]进行池化操作，且 pad_value = 0.0
        输出为 Tensor：
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            其中 out.shape[0] == len(x.lod[-1]) == 5
            sum: out.data = [[1.], [5.], [4.], [0.0], [12.]]
            where 1.=1., 5.=3. + 2., 4.=4., 0.0=pad_value, 12.=6. + 5. + 1.


参数
:::::::::
    - **input** (Tensor) - 类型为 Tensor 的输入序列，仅支持 lod_level 不超过 2 的 Tensor，数据类型为 float32。
    - **pool_type** (str) - 池化类型，支持 average，sum，sqrt，max，last 和 first 池化操作。
    - **is_test** (bool，可选) - 仅在 pool_type 取值为 max 时生效。当 is_test 为 False 时，则在池化操作过程中会创建 maxIndex 临时 Tenosr，以记录最大特征值对应的索引信息，用于训练阶段的反向梯度计算。默认为 False。
    - **pad_value** (float，可选) - 用于填充输入序列为空时的池化结果，默认为 0.0。

返回
:::::::::
经过指定类型池化后的 Tensor，数据类型为 float32。

代码示例
:::::::::
COPY-FROM: paddle.static.nn.sequence_pool
