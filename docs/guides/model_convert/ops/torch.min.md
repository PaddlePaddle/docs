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

### [paddle.min](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/min_cn.html#min)

```python
paddle.min(x, 
            axis=None, 
            keepdim=False, 
            name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| dim           | axis         | 求最小值运算的维度。                                      |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |


### 代码示例
``` python
# PyTorch示例：
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
#        [ 0.2457,  0.0384,  1.0128,  0.7015],
#        [-0.1153,  2.9849,  2.1458,  0.5788]])
torch.min(a, 1)
# 输出
# torch.return_types.min(values=tensor([-1.1899, -1.4644,  0.0384, -0.1153]), indices=tensor([2, 0, 1, 0]))
```

``` python
# PaddlePaddle示例：
# data_x is a Tensor with shape [2, 4]
# the axis is a int element
x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                      [0.1, 0.2, 0.6, 0.7]],
                     dtype='float64', stop_gradient=False)
result1 = paddle.min(x)
result1.backward()
print(result1, x.grad)
# 输出
# [0.1], [[0., 0., 0., 0.], [1., 0., 0., 0.]]

x.clear_grad()
result2 = paddle.min(x, axis=0)
result2.backward()
print(result2, x.grad)
# 输出
# [0.1, 0.2, 0.5, 0.7], [[0., 0., 1., 0.], [1., 1., 0., 1.]]

x.clear_grad()
result3 = paddle.min(x, axis=-1)
result3.backward()
print(result3, x.grad)
# 输出
# [0.2, 0.1], [[1., 0., 0., 0.], [1., 0., 0., 0.]]

x.clear_grad()
result4 = paddle.min(x, axis=1, keepdim=True)
result4.backward()
print(result4, x.grad)
# 输出
# [[0.2], [0.1]], [[1., 0., 0., 0.], [1., 0., 0., 0.]]

# data_y is a Tensor with shape [2, 2, 2]
# the axis is list
y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                      [[5.0, 6.0], [7.0, 8.0]]],
                     dtype='float64', stop_gradient=False)
result5 = paddle.min(y, axis=[1, 2])
result5.backward()
print(result5, y.grad)
# 输出
# [1., 5.], [[[1., 0.], [0., 0.]], [[1., 0.], [0., 0.]]]

y.clear_grad()
result6 = paddle.min(y, axis=[0, 1])
result6.backward()
print(result6, y.grad)
# 输出
# [1., 2.], [[[1., 1.], [0., 0.]], [[0., 0.], [0., 0.]]]
```
