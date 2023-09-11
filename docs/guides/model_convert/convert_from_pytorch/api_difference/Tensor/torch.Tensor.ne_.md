## [ 参数不一致 ] torch.Tensor.ne_
### [torch.Tensor.ne_](https://pytorch.org/docs/stable/generated/torch.Tensor.ne_.html)

```python
torch.Tensor.ne_(other)
```

### [paddle.Tensor.not_equal_]()

```python
paddle.Tensor.not_equal_(y)
```

其中，Paddle 与 PyTorch 的 `other` 参数所支持类型不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                             |
| ------------- | ------------ | ----------------------------------------------- |
| other         | y            | 比较的元素，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要转写。                       |

### 转写示例
#### other：比较的元素
```python
# PyTorch 写法
y = x.ne_(other=2)

# Paddle 写法
y = x.not_equal_(y=paddle.to_tensor(2))
```
