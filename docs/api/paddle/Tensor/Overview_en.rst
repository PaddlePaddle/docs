
.. _en_paddle_Tensor:

``Tensor`` can be regarded as multi-dimensional array, which can have as many diemensions as it want. For more information, you can see `Introduction to Tensor <.https://www.paddlepaddle.org.cn/documentation/docs/en/guides/01_paddle2.0_introduction/basic_concept/tensor_introduction_en.html>`_.

Data types
----------

PaddlePaddle defines some tensor types which are as follows:

======================================= ===========================================
Data type                               dtype                                      
======================================= ===========================================
32-bit floating point                   ``paddle.float32``                         
64-bit floating point                   ``paddle.float64``                         
16-bit floating point                   ``paddle.float16``                         
16-bit floating point                   ``paddle.bfloat16``                         
64-bit complex                          ``paddle.complex64``                         
128-bit complex                         ``paddle.complex128``                         
8-bit integer (unsigned)                ``paddle.uint8``                         
8-bit integer (signed)                  ``paddle.int8``                         
16-bit integer (signed)                 ``paddle.int16``                         
32-bit integer (signed)                 ``paddle.int32``                         
64-bit integer (signed)                 ``paddle.int64``                         
Boolean                                 ``paddle.bool``                         
======================================= ===========================================

Tensor class reference
----------------------

.. currentmodule:: paddle.Tensor


Properties
~~~~~~~~~~~~~~~~~~~~~~

======================================= ===========================================
``T``                                   The transpose of ``Tensor``. See :ref:`paddle.transpose <api_paddle_transpose>`                             
``block``                               Tensor's block.
``dtype``                               Tensor's data type.
``grad``                                The value of Tensor's grad.
``inplace_version``                     The inplace version of current Tensor.
``is_leaf``                             Whether Tensor is leaf Tensor.
``name``                                The name of Tensor.
``ndim``                                The dimensions of Tensor.
``persistable``                         The value of Tensor's persistable.
``place``                               The place of Tensor.
``shape``                               The shape of Tensor. See :ref:`paddle.shape <api_paddle_shape>`
``size``                                The size of Tensor. See :ref:`paddle.numel <api_paddle_numel>`
``stop_gradient``                       The value of Tensor's stop_gradient.
``type``                                Tensor's type.
======================================= ===========================================


Methods
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    abs
    acos
    add
    add_
    add_n
    addmm
    all
    allclose
    any
    argmax
    argmin
    argsort
    asin
    astype
    atan
    backward
    bitwise_and
    bitwise_not
    bitwise_or
    bitwise_xor
    bmm
    broadcast_shape
    broadcast_tensors
    broadcast_to
    cast
    ceil
    ceil_
    cholesky
    chunk
    clear_grad
    clear_gradient
    clip
    clip_
    clone
    concat
    cond
    conj
    copy_
    cos
    cosh
    cpu
    cross
    cuda
    cumprod
    cumsum
    detach
    diagonal
    digamma
    dim
    dist
    divide
    dot
    eigvals
    equal
    equal_all
    erf
    exp
    exp_
    expand
    expand_as
    fill_
    fill_diagonal_
    fill_diagonal_tensor
    fill_diagonal_tensor_
    flatten
    flatten_
    flip
    floor
    floor_
    floor_divide
    floor_mod
    gather
    gather_nd
    gradient
    greater_equal
    greater_than
    histogram
    imag
    increment
    index_sample
    index_select
    inverse
    is_empty
    is_tensor
    isfinite
    isinf
    isnan
    item
    kron
    less_equal
    less_than
    lgamma
    log
    log10
    log1p
    log2
    logical_and
    logical_not
    logical_or
    logical_xor
    logsumexp
    masked_select
    matmul
    matrix_power
    max
    maximum
    mean
    median
    min
    minimum
    mm
    mod
    multiplex
    multiply
    mv
    ndimension
    neg
    nonzero
    norm
    not_equal
    numel
    numpy
    pin_memory
    pow
    prod
    rank
    real
    reciprocal
    reciprocal_
    register_hook
    remainder
    reshape
    reshape_
    reverse
    roll
    round
    round_
    rsqrt
    rsqrt_
    scale
    scale_
    scatter
    scatter_
    scatter_nd
    scatter_nd_add
    set_value
    shard_index
    sign
    sin
    sinh
    slice
    solve
    sort
    split
    sqrt
    sqrt_
    square
    squeeze
    squeeze_
    stack
    stanh
    std
    strided_slice
    subtract
    subtract_
    sum
    t
    tanh
    tanh_
    tile
    tolist
    topk
    trace
    transpose
    trunc
    unbind
    uniform_
    unique
    unique_consecutive
    unsqueeze
    unsqueeze_
    unstack
    value
    var
    where
    zero_
