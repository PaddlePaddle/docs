.. _user_guide_broadcasting:

==================
Broadcasting
==================

PaddlePaddle provides broadcasting semantics in some APIs like other deep learning frameworks, which allows using tensors with different shapes while operating.
In General, broadcast is the rule how the smaller tensor is “broadcast” across the larger tsnsor so that they have same shapes.
Note that no copies happened while broadcasting.  

In Paddlepaddle, tensors are broadcastable when following rulrs hold(ref: `Numpy Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html#module-numpy.doc.broadcasting>`_ ):

1. there should be at least one dimention in each tensor
2. when we compare their shapes element-wise from backward to forward, two dimensions are compatible when 
they are equal, or one of them is 1, or one of them does not exist.

For example:

.. code-block:: python

    import paddle
    
    x = paddle.ones((2, 3, 4))
    y = paddle.ones((2, 3, 4))
    # Two tensor have some shpes are broadcastable
    z = x + y
    print(z.shape) 
    # [2, 3, 4]
    
    x = paddle.ones((2, 3, 1, 5))
    y = paddle.ones((3, 4, 1))

    # compare from backward to forward：
    # 1st step：y's dimention is 1
    # 2nd step：x's dimention is 1
    # 3rd step：two dimentions are the same
    # 4st step：y's dimention does not exist
    # So, x and y are broadcastable
    z = x + y
    print(z.shape) 
    # [2, 3, 4, 5]

    # In Compare
    x = paddle.ones((2, 3, 4))
    y = paddle.ones((2, 3, 6))
    # x and y are not broadcastable because in first step form tail, x's dimention 4 is not equal to y's dimention 6
    # z = x, y
    # InvalidArgumentError: Broadcast dimension mismatch.

Now we know in what condition two tensors are broadcastable, how to calculate the resulting tensor's size follows the rules:

1. If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
2. Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension.

For example:

.. code-block:: python

    import paddle

    x = paddle.ones((2, 1, 4))
    y = paddle.ones((3, 1))
    z = x + y
    print(z.shape)
    # z'shape: [2, 3, 4]

    x = paddle.ones((2, 1, 4))
    y = paddle.ones((3, 2))
    z = x + y
    print(z.shape)
    # InvalidArgumentError: Broadcast dimension mismatch.
