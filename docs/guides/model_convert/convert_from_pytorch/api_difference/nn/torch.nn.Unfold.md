## torch.nn.Unfold
### [torch.nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=nn+unfold#torch.nn.Unfold)

```python
torch.nn.Unfold(kernel_size,
                dilation=1,
                padding=0,
                stride=1)
```

### [paddle.nn.Unfold](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Unfold_cn.html#unfold)

```python
paddle.nn.Unfold(kernel_size=[3, 3],
                    strides=1,
                    paddings=1,
                    dilation=1,
                    name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding       | paddings     | 每个维度的扩展，整数或者整型列表。                   |
| stride        | strides      | 卷积步长，整数或者整型列表。                        |


### 功能差异

#### 使用方式
***PyTorch***：前四个参数类型为 int、tuple(int) 或者 list(int)。
***PaddlePaddle***：前四个参数类型为 int 或者 list(int)。


### 代码示例
``` python
# PyTorch 示例：
unfold = nn.Unfold(kernel_size=(2, 3))
input = torch.randn(2, 5, 3, 4)
output = unfold(input)
# each patch contains 30 values (2x3=6 vectors, each of 5 channels)
# 4 blocks (2x3 kernels) in total in the 3x4 input
output.size()
# 输出
# torch.Size([2, 30, 4])

# Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
inp = torch.randn(1, 3, 10, 12)
w = torch.randn(2, 3, 4, 5)
inp_unf = torch.nn.functional.unfold(inp, (4, 5))
out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
# or equivalently (and avoiding a copy),
# out = out_unf.view(1, 2, 7, 8)
(torch.nn.functional.conv2d(inp, w) - out).abs().max()
# 输出
# tensor(1.9073e-06)
```

``` python
# PaddlePaddle 示例：
b = paddle.arange(10).reshape([5,2])
# 输出
# Tensor(shape=[5, 2], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[0, 1],
#         [2, 3],
#         [4, 5],
#         [6, 7],
#         [8, 9]])
paddle.split(b, 2, 1)
# 输出
# [Tensor(shape=[5, 1], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[0],
#         [2],
#         [4],
#         [6],
#         [8]]), Tensor(shape=[5, 1], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[1],
#         [3],
#         [5],
#         [7],
#         [9]])]
```
