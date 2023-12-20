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
    count_nonzero
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
    diff
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
    isclose
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
    logcumsumexp
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
    fmax
    mean
    median
    nanmedian
    min
    minimum
    fmin
    mm
    inner
    outer
    cov
    lu
    lu_unpack
    cholesky_solve
    mod
    multiplex
    multiply
    mv
    nan_to_num
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
    repeat_interleave
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
    sgn
    sin
    sinh
    slice
    solve
    sort
    split
    vsplit
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
    trunc
    frac
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
    is_complex
    is_integer
    frexp
    trapezoid
    cumulative_trapezoid
    vander
    hypot
    hypot_
    diagonal_scatter
    combinations
