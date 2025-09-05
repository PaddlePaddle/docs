.. _en_paddle_Tensor:

paddle.Tensor
========================

A ``Tensor`` is a generalization of vectors and matrices and is easily understood as a multidimensional array. For more information, you can see `Introduction to Tensor <https://www.paddlepaddle.org.cn/documentation/docs/en/guides/01_paddle2.0_introduction/basic_concept/tensor_introduction_en.html>`_.

Data types
----------

PaddlePaddle defines the following Tensor types:

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
``T``                                   The transpose of ``Tensor``. See :ref:`paddle.transpose <api_paddle_transpose>` .
``mT``                                  Matrix transpose of ``Tensor`` (swap the last two dimensions). See :ref:`paddle.matrix_transpose <api_paddle_matrix_transpose>`.
``block``                               Tensor's block.
``dtype``                               Tensor's data type.
``grad``                                The value of Tensor's grad.
``inplace_version``                     The inplace version of current Tensor.
``is_leaf``                             Whether Tensor is leaf Tensor.
``name``                                The name of Tensor.
``ndim``                                The dimensions of Tensor.
``persistable``                         The value of Tensor's persistable.
``place``                               The place of Tensor.
``shape``                               The shape of Tensor. See :ref:`paddle.shape <api_paddle_shape>` .
``size``                                The size of Tensor. See :ref:`paddle.numel <api_paddle_numel>` .
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
    angle
    any
    argmax
    argmin
    argsort
    as_complex
    as_real
    asin
    astype
    atan
    backward
    bernoulli_
    bitwise_and
    bitwise_not
    bitwise_or
    bitwise_xor
    bmm
    broadcast_shape
    broadcast_tensors
    broadcast_to
    bucketize
    cast
    ceil
    ceil_
    cholesky
    cholesky_solve
    chunk
    clear_grad
    clear_gradient
    clip
    clip_
    clone
    combinations
    concat
    cond
    conj
    copy_
    cos
    cosh
    count_nonzero
    cov
    cpu
    cross
    cuda
    cumprod
    cumsum
    cumulative_trapezoid
    dense_dim
    detach
    diagonal
    diagonal_scatter
    diff
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
    fmax
    fmin
    frac
    frexp
    gather
    gather_nd
    gradient
    greater_equal
    greater_than
    histogram
    histogram_bin_edges
    hypot
    hypot_
    imag
    increment
    index_sample
    index_select
    inner
    inverse
    is_coalesced
    is_complex
    is_empty
    is_integer
    is_tensor
    isclose
    isfinite
    isin
    isinf
    isnan
    isneginf
    isposinf
    isreal
    item
    kron
    less_equal
    less_than
    lgamma
    log
    log_normal_
    log10
    log1p
    log2
    logcumsumexp
    logical_and
    logical_not
    logical_or
    logical_xor
    logsumexp
    lu
    lu_solve
    lu_unpack
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
    nan_to_num
    nanmedian
    ndimension
    neg
    new_full
    new_ones
    new_zeros
    new_empty
    nonzero
    norm
    not_equal
    numel
    numpy
    outer
    pin_memory
    pow
    prod
    rank
    real
    reciprocal
    reciprocal_
    register_hook
    remainder
    repeat_interleave
    reshape
    reshape_
    resize_
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
    set_
    set_value
    sgn
    shard_index
    sign
    sin
    sinc
    sinc_
    sinh
    slice
    solve
    sort
    sparse_dim
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
    take
    tanh
    tanh_
    tile
    to
    tolist
    topk
    trace
    transpose
    trapezoid
    trunc
    unbind
    uniform_
    unique
    unique_consecutive
    unsqueeze
    unsqueeze_
    unstack
    value
    vander
    var
    vsplit
    where
    zero_
