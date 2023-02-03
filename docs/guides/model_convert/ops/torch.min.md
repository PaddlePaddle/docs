## torch.min
### [torch.min](https://pytorch.org/docs/stable/generated/torch.min.html?highlight=min#torch.min)

```python
torch.min(input)
```

```python
torch.min(input,
            dim,
            keepdim=False,
            *,
            out=None)
```

```python
torch.min(input,
            other,
            *,
            out=None)
```

### [paddle.min](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/min_cn.html#min)

```python
paddle.min(x,
            axis=None,
            keepdim=False,
            name=None)
```
### [paddle.minimum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/minimum_cn.html#minimum)

```python
paddle.minimum(x,
                y,
                name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| dim           | axis         | 求最小值运算的维度。                                      |
| other         | y            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 功能差异

#### 使用方式
***PyTorch***：在输入 dim 时，返回 (values, indices)；在输入 other 时，比较 input 和 other 返回较小值。
***PaddlePaddle***：paddle.min 对指定维度上的 Tensor 元素求最小值运算，返回 Tensor；paddle.minimum 逐元素对比输入的两个 Tensor，返回 Tensor。


### 代码示例
``` python
# PyTorch 示例：
a = torch.randn(1, 3)
a
# 输出
# tensor([[ 0.6750,  1.0857,  1.7197]])
torch.min(a)
# 输出
# tensor(0.6750)

a = torch.randn(4, 4)
a
# 输出
# tensor([[-0.6248,  1.1334, -1.1899, -0.2803],
#        [-1.4644, -0.2635, -0.3651,  0.6134],
#        [ 0.2457,  0.0384,  1.0128,  0.7015],
#        [-0.1153,  2.9849,  2.1458,  0.5788]])
torch.min(a, 1)
# 输出
# torch.return_types.min(values=tensor([-1.1899, -1.4644,  0.0384, -0.1153]), indices=tensor([2, 0, 1, 0]))

a = torch.tensor((1, 2, -1))
b = torch.tensor((3, 0, 4))
torch.minimum(a, b)
# 输出
# tensor([1, 0, -1])
```

``` python
# PaddlePaddle 示例：
# paddle.min
# data_x is a Tensor with shape [2, 4]
# the axis is a int element
x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                      [0.1, 0.2, 0.6, 0.7]],
                     dtype='float64', stop_gradient=False)
result1 = paddle.min(x)
result1.backward()
print(result1, x.grad)
# 输出
#[0.1], [[0., 0., 0., 0.], [1., 0., 0., 0.]]

x.clear_grad()
result2 = paddle.min(x, axis=0)
result2.backward()
print(result2, x.grad)
# 输出
#[0.1, 0.2, 0.5, 0.7], [[0., 0., 1., 0.], [1., 1., 0., 1.]]

x.clear_grad()
result3 = paddle.min(x, axis=-1)
result3.backward()
print(result3, x.grad)
# 输出
#[0.2, 0.1], [[1., 0., 0., 0.], [1., 0., 0., 0.]]

x.clear_grad()
result4 = paddle.min(x, axis=1, keepdim=True)
result4.backward()
print(result4, x.grad)
# 输出
#[[0.2], [0.1]], [[1., 0., 0., 0.], [1., 0., 0., 0.]]

# data_y is a Tensor with shape [2, 2, 2]
# the axis is list
y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                      [[5.0, 6.0], [7.0, 8.0]]],
                     dtype='float64', stop_gradient=False)
result5 = paddle.min(y, axis=[1, 2])
result5.backward()
print(result5, y.grad)
# 输出
#[1., 5.], [[[1., 0.], [0., 0.]], [[1., 0.], [0., 0.]]]

y.clear_grad()
result6 = paddle.min(y, axis=[0, 1])
result6.backward()
print(result6, y.grad)
# 输出
#[1., 2.], [[[1., 1.], [0., 0.]], [[0., 0.], [0., 0.]]]

# paddle.minimum
x = paddle.to_tensor([[1, 2], [7, 8]])
y = paddle.to_tensor([[3, 4], [5, 6]])
res = paddle.minimum(x, y)
print(res)
# 输出
# Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[1, 2],
#         [5, 6]])

x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
y = paddle.to_tensor([3, 0, 4])
res = paddle.minimum(x, y)
print(res)
# 输出
# Tensor(shape=[1, 2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[[1, 0, 3],
#          [1, 0, 3]]])

x = paddle.to_tensor([2, 3, 5], dtype='float32')
y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
res = paddle.minimum(x, y)
print(res)
# 输出
# Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [1. , nan, nan])

x = paddle.to_tensor([5, 3, float("inf")], dtype='float64')
y = paddle.to_tensor([1, -float("inf"), 5], dtype='float64')
res = paddle.minimum(x, y)
print(res)
# 输出
# Tensor(shape=[3], dtype=float64, place=Place(cpu), stop_gradient=True,
#        [ 1.  , -inf.,  5.  ])
```
