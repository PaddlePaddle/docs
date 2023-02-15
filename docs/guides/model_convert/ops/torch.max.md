## torch.max
### [torch.max](https://pytorch.org/docs/stable/generated/torch.max.html?highlight=max#torch.max)

```python
torch.max(input)
```

```python
torch.max(input,
            dim,
            keepdim=False,
            *,
            out=None)
```

```python
torch.max(input,
            other,
            *,
            out=None)
```

### [paddle.max](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/max_cn.html#max)

```python
paddle.max(x,
            axis=None,
            keepdim=False,
            name=None)
```

### [paddle.maximum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/maximum_cn.html#maximum)

```python
paddle.maximum(x,
                y,
                name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| dim           | axis         | 求最大值运算的维度。                                      |
| other         | y            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 功能差异

#### 使用方式
***PyTorch***：在输入 dim 时，返回 (values, indices)；在输入 other 时，比较 input 和 other 返回较大值。
***PaddlePaddle***：paddle.max 对指定维度上的 Tensor 元素求最大值运算，返回 Tensor；paddle.maximum 逐元素对比输入的两个 Tensor，返回 Tensor。


### 代码示例
``` python
# PyTorch 示例：
a = torch.randn(1, 3)
a
# 输出
# tensor([[ 0.6763,  0.7445, -2.2369]])
torch.max(a)
# 输出
# tensor(0.7445)

a = torch.randn(4, 4)
a
# 输出
# tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
#        [ 1.1949, -1.1127, -2.2379, -0.6702],
#        [ 1.5717, -0.9207,  0.1297, -1.8768],
#        [-0.6172,  1.0036, -0.6060, -0.2432]])
torch.max(a, 1)
# 输出
# torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))

a = torch.tensor((1, 2, -1))
b = torch.tensor((3, 0, 4))
torch.max(a, b)
# 输出
# tensor([3, 2, 4])
```

``` python
# PaddlePaddle 示例：
# paddle.max
# data_x is a Tensor with shape [2, 4]
# the axis is a int element
x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                      [0.1, 0.2, 0.6, 0.7]],
                     dtype='float64', stop_gradient=False)
result1 = paddle.max(x)
result1.backward()
print(result1, x.grad)
# 输出
#[0.9], [[0., 0., 0., 1.], [0., 0., 0., 0.]]

x.clear_grad()
result2 = paddle.max(x, axis=0)
result2.backward()
print(result2, x.grad)
# 输出
#[0.2, 0.3, 0.6, 0.9], [[1., 1., 0., 1.], [0., 0., 1., 0.]]

x.clear_grad()
result3 = paddle.max(x, axis=-1)
result3.backward()
print(result3, x.grad)
# 输出
#[0.9, 0.7], [[0., 0., 0., 1.], [0., 0., 0., 1.]]

x.clear_grad()
result4 = paddle.max(x, axis=1, keepdim=True)
result4.backward()
print(result4, x.grad)
# 输出
#[[0.9], [0.7]], [[0., 0., 0., 1.], [0., 0., 0., 1.]]

# data_y is a Tensor with shape [2, 2, 2]
# the axis is list
y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                      [[5.0, 6.0], [7.0, 8.0]]],
                     dtype='float64', stop_gradient=False)
result5 = paddle.max(y, axis=[1, 2])
result5.backward()
print(result5, y.grad)
# 输出
#[4., 8.], [[[0., 0.], [0., 1.]], [[0., 0.], [0., 1.]]]

y.clear_grad()
result6 = paddle.max(y, axis=[0, 1])
result6.backward()
print(result6, y.grad)
# 输出
#[7., 8.], [[[0., 0.], [0., 0.]], [[0., 0.], [1., 1.]]]

# paddle.maximum
x = paddle.to_tensor([[1, 2], [7, 8]])
y = paddle.to_tensor([[3, 4], [5, 6]])
res = paddle.maximum(x, y)
print(res)
# 输出
# Tensor(shape=[2, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[3, 4],
#         [7, 8]])

x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
y = paddle.to_tensor([3, 0, 4])
res = paddle.maximum(x, y)
print(res)
# 输出
# Tensor(shape=[2, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
#        [[3, 2, 4],
#         [3, 2, 4]])

x = paddle.to_tensor([2, 3, 5], dtype='float32')
y = paddle.to_tensor([1, float("nan"), float("nan")], dtype='float32')
res = paddle.maximum(x, y)
print(res)
# 输出
# Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [2. , nan, nan])

x = paddle.to_tensor([5, 3, float("inf")], dtype='float32')
y = paddle.to_tensor([1, -float("inf"), 5], dtype='float32')
res = paddle.maximum(x, y)
print(res)
# 输出
# Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
#        [5.  , 3.  , inf.])
```
