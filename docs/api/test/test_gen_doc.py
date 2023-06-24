import unittest
from parameterized import parameterized

from gen_doc import extract_code_blocks_from_docstr

docstrings = [
    (
        "**Scatter Layer**\nOutput is obtained by updating the input on selected indices based on updates.\n\n.. code-block:: python\n\n    import numpy as np\n    #input:\n    x = np.array([[1, 1], [2, 2], [3, 3]])\n    index = np.array([2, 1, 0, 1])\n    # shape of updates should be the same as x\n    # shape of updates with dim > 1 should be the same as input\n    updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])\n    overwrite = False\n    # calculation:\n    if not overwrite:\n        for i in range(len(index)):\n            x[index[i]] = np.zeros((2))\n    for i in range(len(index)):\n        if (overwrite):\n            x[index[i]] = updates[i]\n        else:\n            x[index[i]] += updates[i]\n    # output:\n    out = np.array([[3, 3], [6, 6], [1, 1]])\n    out.shape # [3, 2]\n\n**NOTICE**: The order in which updates are applied is nondeterministic,\nso the output will be nondeterministic if index contains duplicates.\n\nArgs:\n    x (Tensor): The input N-D Tensor with ndim>=1. Data type can be float32, float64.\n    index (Tensor): The index 1-D Tensor. Data type can be int32, int64. The length of index cannot exceed updates's length, and the value in index cannot exceed input's length.\n    updates (Tensor): update input with updates parameter based on index. shape should be the same as input, and dim value with dim > 1 should be the same as input.\n    overwrite (bool): The mode that updating the output when there are same indices.\n\n        If True, use the overwrite mode to update the output of the same index,\n            if False, use the accumulate mode to update the output of the same index.Default value is True.\n\n    name(str, optional): The default value is None. Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .\n\nReturns:\n    Tensor: The output is a Tensor with the same shape as x.\n\nExamples:\n    .. code-block:: python\n\n        import paddle\n\n        x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')\n        index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')\n        updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')\n\n        output1 = paddle.scatter(x, index, updates, overwrite=False)\n        # [[3., 3.],\n        #  [6., 6.],\n        #  [1., 1.]]\n\n        output2 = paddle.scatter(x, index, updates, overwrite=True)\n        # CPU device:\n        # [[3., 3.],\n        #  [4., 4.],\n        #  [1., 1.]]\n        # GPU device maybe have two results because of the repeated numbers in index\n        # result 1:\n        # [[3., 3.],\n        #  [4., 4.],\n        #  [1., 1.]]\n        # result 2:\n        # [[3., 3.],\n        #  [2., 2.],\n        #  [1., 1.]]",
        """import paddle

x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

output1 = paddle.scatter(x, index, updates, overwrite=False)
# [[3., 3.],
#  [6., 6.],
#  [1., 1.]]

output2 = paddle.scatter(x, index, updates, overwrite=True)
# CPU device:
# [[3., 3.],
#  [4., 4.],
#  [1., 1.]]
# GPU device maybe have two results because of the repeated numbers in index
# result 1:
# [[3., 3.],
#  [4., 4.],
#  [1., 1.]]
# result 2:
# [[3., 3.],
#  [2., 2.],
#  [1., 1.]]""",
    ),
    (
        "Output is obtained by gathering entries of ``axis``\nof ``x`` indexed by ``index`` and concatenate them together.\n\n.. code-block:: python\n\n\n            Given:\n\n            x = [[1, 2],\n                 [3, 4],\n                 [5, 6]]\n\n            index = [1, 2]\n            axis=[0]\n\n            Then:\n\n            out = [[3, 4],\n                   [5, 6]]\n\nArgs:\n    x (Tensor): The source input tensor with rank>=1. Supported data type is\n        int32, int64, float32, float64 and uint8 (only for CPU),\n        float16 (only for GPU).\n    index (Tensor): The index input tensor with rank=1. Data type is int32 or int64.\n    axis (Tensor|int, optional): The axis of input to be gathered, it's can be int or a Tensor with data type is int32 or int64. The default value is None, if None, the ``axis`` is 0.\n    name (str, optional): The default value is None.  Normally there is no need for user to set this property.\n        For more information, please refer to :ref:`api_guide_Name` .\n\nReturns:\n    output (Tensor): The output is a tensor with the same rank as ``x``.\n\nExamples:\n\n    .. code-block:: python\n\n        import paddle\n\n        input = paddle.to_tensor([[1,2],[3,4],[5,6]])\n        index = paddle.to_tensor([0,1])\n        output = paddle.gather(input, index, axis=0)\n        # expected output: [[1,2],[3,4]]",
        """import paddle

input = paddle.to_tensor([[1,2],[3,4],[5,6]])
index = paddle.to_tensor([0,1])
output = paddle.gather(input, index, axis=0)
# expected output: [[1,2],[3,4]]""",
    ),
    (
        "Examples:\n    .. code-block:: python\n\n        import paddle\n\n        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])\n        out = paddle.abs(x)\n        print(out)\n        # [0.4 0.2 0.1 0.3]\n",
        """import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.abs(x)
print(out)
# [0.4 0.2 0.1 0.3]""",
    ),
    (
        "\nExamples:\n    .. code-block:: python\n\n        >>> import paddle\n        >>> x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])\n        >>> out = paddle.abs(x)\n        >>> print(out)\n        Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True, [0.40000001, 0.20000000, 0.10000000, 0.30000001])\n\n",
        """import paddle
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.abs(x)
print(out)
# Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True, [0.40000001, 0.20000000, 0.10000000, 0.30000001])""",
    ),
]


class StripPS1Test(unittest.TestCase):
    @parameterized.expand(docstrings)
    def test_strip_ps1_from_codeblock(self, docstring, target):
        codeblocks = extract_code_blocks_from_docstr(docstring)
        codeblock = [
            codeblock
            for codeblock in codeblocks
            if codeblock.get("in_examples")
        ][0]["codes"]

        assert codeblock == target


if __name__ == '__main__':
    unittest.main()
