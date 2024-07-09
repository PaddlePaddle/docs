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

Args:
    query (Tensor): The query tensor in the attention module.
                    A 4-D tensor with shape [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
    key (Tensor): The key tensor in the attention module.
                    A 4-D tensor with shape [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
    value (Tensor): The value tensor in the attention module.
                    A 4-D tensor with shape [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
    startend_row_indices (Tensor): A sparse attention mask indices tensor.
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
    dropout (float): The dropout ratio. Default is 0.0.
    causal (bool): Whether to enable causal mode. Default is False.
    return_softmax_lse (bool): Whether to return the log-sum-exp of the softmax. Default is False.
    return_seed_offset (bool): Whether to return the random seed offset. Default is False.
    fixed_seed_offset (Tensor, optional): With fixed seed, offset for dropout mask.
    rng_name (str): The name to select Generator.
    training (bool): Whether the module is in training mode. Default is True.
    name (str, optional): Name of the operation. Default is None. Normally, users do not need to set this property.
                            For more information, refer to :ref:`api_guide_Name`.

Returns:
    Tensor: The computed attention result with the same shape as the input `value`.
