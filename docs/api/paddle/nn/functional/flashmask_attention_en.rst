Implements FlashAttention with a sparse mask representation.

The equation is:

.. math::

    result = softmax(\frac{Q \cdot K^T}{\sqrt{d}} + M) \cdot V

where ``Q``, ``K``, and ``V`` are the input tensors of the attention module.
They share the same dimensions, and ``d`` represents the size of the last dimension.
``M`` is the dense mask.

The figure below shows examples of various masks, with the Score matrix depicted. Gray areas indicate elements that are masked. The numbers above represent the values of `startend_row_indices`. A single row of numbers indicates that the shape of `startend_row_indices` is `[batch_size, num_heads, seq_len, 1]`. Two rows of numbers indicate that the shape of `startend_row_indices` is `[batch_size, num_heads, seq_len, 2]`. Four rows of numbers indicate that the shape of `startend_row_indices` is `[batch_size, num_heads, seq_len, 4]`.

.. image:: https:/githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/FlashMask1.png
    :width: 900
    :alt: pipeline
    :align: center

In Figure (a), where `causal=True`, the values of `startend_row_indices` are as follows:

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
        [[[[5 ],
            [5 ],
            [5 ],
            [5 ],
            [5 ],
            [5 ],
            [5 ],
            [5],
            [5],
            [5]]]])

In Figure (b), where `causal=True`, the values of `startend_row_indices` are as follows:

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
        [[[[4 ],
            [4 ],
            [4 ],
            [4 ],
            [7 ],
            [7 ],
            [7 ],
            [10],
            [10],
            [10]]]])

.. image:: https:/githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/FlashMask2.png
    :width: 900
    :alt: pipeline
    :align: center

In Figure (c), where `causal=True`, the values of `startend_row_indices` are as follows:

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
        [[[[10 ],
            [10 ],
            [10 ],
            [10 ],
            [7 ],
            [7 ],
            [7 ],
            [10],
            [10],
            [10]]]])

In Figure (d), where `causal=True`, the values of `startend_row_indices` are as follows:

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
        [[[[10 ],
            [4 ],
            [5 ],
            [6 ],
            [7 ],
            [8 ],
            [9 ],
            [10],
            [10],
            [10]]]])

.. image:: https:/githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/FlashMask3.png
    :width: 900
    :alt: pipeline
    :align: center

In Figure (e), where `causal=True`, the values of `startend_row_indices` are as follows:

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
        [[[[4 , 7 ],
            [4 , 7 ],
            [4 , 7 ],
            [4 , 7 ],
            [10, 10],
            [10, 10],
            [10, 10],
            [10, 10],
            [10, 10],
            [10, 10]]]])

In Figure (f), where `causal=False`, the values of `startend_row_indices` are as follows:

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
        [[[[4 , 0 ],
            [4 , 0 ],
            [4 , 0 ],
            [4 , 0 ],
            [7, 4],
            [7, 4],
            [7, 4],
            [10, 7],
            [10, 7],
            [10, 7]]]])

.. image:: https:/githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/images/FlashMask4.png
    :width: 900
    :alt: pipeline
    :align: center

In Figure (g), where `causal=False`, the values of `startend_row_indices` are as follows:

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 4], dtype=int32, place=Place(gpu:0), stop_gradient=True,
        [[[[10, 10, 0 , 0 ],
            [10, 10, 0 , 0 ],
            [10, 10, 0 , 0 ],
            [3 , 10, 0 , 0 ],
            [4 , 10, 3 , 4 ],
            [5 , 10, 3 , 5 ],
            [6 , 10, 3 , 6 ],
            [7 , 10, 3 , 7 ],
            [8 , 10, 3 , 8 ],
            [9 , 10, 3 , 9 ]]]])

In Figure (h), where `causal=True`, the values of `startend_row_indices` are as follows:

.. code-block:: python

    >>> print(startend_row_indices)
    Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
        [[[[10 ],
            [4 ],
            [8 ],
            [6 ],
            [10 ],
            [7 ],
            [10 ],
            [9],
            [10],
            [10]]]])

Warning:
    This API only supports inputs with dtype float16 and bfloat16.

Args
::::::::::::
    - **query** (Tensor) - The query tensor in the attention module.
                    A 4-D tensor with shape [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
    - **key** (Tensor) - The key tensor in the attention module.
                    A 4-D tensor with shape [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
    - **value** (Tensor) - The value tensor in the attention module.
                    A 4-D tensor with shape [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
    - **startend_row_indices** (Tensor)
        - A sparse attention mask indices tensor.
            A 4-D tensor with shape [batch_size, num_heads, seq_len, {1, 2, 4}].
            The dtype must be int32. num_heads can be 1 or the same as key's num_heads. When num_heads is 1, it will be broadcast to match key's num_heads.
            Depending on the value of the causal parameter, startend_row_indices can take different shapes and meanings, with the values in startend_row_indices being denoted as r1, r2, r3, r4 sequentially.
        - When `causal=True` and the shape is [batch_size, num_heads, seq_len, 1],
            indicating unidirectional attention. The value represents the starting row index of the left
            lower triangular mask in the dense mask. The value r1 in startend_row_indices indicates that elements in the lower left triangle of the Score matrix starting from the r1-th row downwards (inclusive) will be masked.
        - When `causal=True` and the shape is [batch_size, num_heads, seq_len, 2],
            indicating unidirectional attention. The values represent the starting and ending row indices of
            the left lower triangular mask in the dense mask. The values r1, r2 in startend_row_indices indicate that elements in the lower left triangle of the Score matrix starting from the r1-th row downwards (inclusive) but above the r2-th row (exclusive) will be masked.
        - When `causal=False` and the shape is [batch_size, num_heads, seq_len, 2],
            indicating bidirectional attention. The values represent the starting row index of the left
            lower triangular mask and the ending row index of the right upper triangular mask in the dense mask. The values r1, r2 in startend_row_indices indicate that elements in the lower left triangle of the Score matrix starting from the r1-th row downwards (inclusive) will be masked, and elements in the upper right triangle starting from the r2-th row upwards (exclusive) will be masked.
        - When `causal=False` and the shape is [batch_size, num_heads, seq_len, 4] (not implemented),
            indicating bidirectional attention. The values represent the start and end row indices of the
            left lower triangular mask and the start and end row indices of the right upper triangular mask in the dense mask. The values r1, r2, r3, r4 in startend_row_indices indicate that elements in the lower left triangle of the Score matrix starting from the r1-th row downwards (inclusive) but above the r2-th row (exclusive) will be masked, and elements in the upper right triangle starting from the r3-th row downwards (inclusive) but above the r4-th row (exclusive) will be masked.
    - **dropout** (float) - The dropout ratio. Default is 0.0.
    - **causal** (bool) - Whether to enable causal mode. Default is False.
    - **return_softmax_lse** (bool) - Whether to return the log-sum-exp of the softmax. Default is False.
    - **return_seed_offset** (bool) - Whether to return the random seed offset. Default is False.
    - **fixed_seed_offset** (Tensor, optional): With fixed seed, offset for dropout mask.
    - **rng_name** (str) - The name to select Generator.
    - **training** (bool) - Whether the module is in training mode. Default is True.
    - **name** (str, optional) - Name of the operation. Default is None. Normally, users do not need to set this property.
                            For more information, refer to :ref:`api_guide_Name` .

Returns
::::::::::::
    Tensor: The computed attention result with the same shape as the input `value`.

Examples:
    .. code-block:: python

        >>> # doctest: +SKIP('flash_attn need A100 compile')
        >>> import paddle

        >>> paddle.seed(2023)
        >>> q = paddle.rand((1, 128, 2, 32),dtype="float16")
        >>> startend_row_indices = paddle.randint(0, 128, (1, 2, 128, 1), dtype="int32")
        >>> output = paddle.nn.functional.flashmask_attention(q, q, q, startend_row_indices, causal=True)
        >>> print(output)
        Tensor(shape=[1, 128, 2, 32], dtype=float16, place=Place(gpu:0), stop_gradient=True,
       [[[[0.81201172, 0.99609375, 0.51074219, ..., 0.80126953,
           0.07232666, 0.83496094],
          [0.34838867, 0.44970703, 0.56103516, ..., 0.68164062,
           0.10986328, 0.07733154]],

         [[0.68603516, 0.85253906, 0.51074219, ..., 0.72119141,
           0.37426758, 0.44531250],
          [0.20300293, 0.79833984, 0.81738281, ..., 0.87890625,
           0.68994141, 0.58496094]],

         [[0.39990234, 0.57080078, 0.40942383, ..., 0.87158203,
           0.14978027, 0.77343750],
          [0.18750000, 0.79443359, 0.76904297, ..., 0.86865234,
           0.76171875, 0.61035156]],

         ...,

         [[0.29321289, 0.67675781, 0.47143555, ..., 0.36621094,
           0.61035156, 0.35668945],
          [0.45825195, 0.21228027, 0.72949219, ..., 0.77246094,
           0.41723633, 0.41870117]],

         [[0.76660156, 0.55322266, 0.73876953, ..., 0.26416016,
           0.63769531, 0.55810547],
          [0.69677734, 0.59863281, 0.77783203, ..., 0.64599609,
           0.36059570, 0.42919922]],

         [[0.31030273, 0.91064453, 0.71826172, ..., 0.29125977,
           0.34423828, 0.60986328],
          [0.73583984, 0.84619141, 0.96728516, ..., 0.61816406,
           0.07440186, 0.55224609]]]])
        >>> # doctest: -SKIP
