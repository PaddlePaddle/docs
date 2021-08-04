
.. _en_paddle_Tensor:


Data types
----------

PaddlePaddle defines some tensor types with CPU and GPU variants which are as follows:

======================================= ===========================================
Data type                               dtype                                      
======================================= ===========================================
32-bit floating point                   ``paddle.float32``                         
======================================= ===========================================


Tensor class reference
----------------------

.. currentmodule:: paddle.Tensor


.. autosummary::
    :nosignatures:

    block
    dtype
    grad
    inplace_version
    is_leaf
    name
    ndim
    persistable
    place
    shape
    size
    stop_gradient
    type

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
    bmm
    broadcast_shape
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
    conj
    copy_
    cos
    cosh
    cpu
    cross
    cuda
    cumsum
    detach
    dim
    dist
    divide
    dot
    equal
    equal_all
    erf
    exp
    exp_
    expand
    expand_as
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
    size
    slice
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
    unbind
    unique
    unsqueeze
    unsqueeze_
    unstack
    value
    var
    where