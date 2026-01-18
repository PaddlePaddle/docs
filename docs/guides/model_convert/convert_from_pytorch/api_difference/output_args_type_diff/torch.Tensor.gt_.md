## [ 返回参数类型不一致 ]torch.Tensor.gt_
### [torch.Tensor.gt_](https://pytorch.org/docs/stable/generated/torch.Tensor.gt_.html)
```python
torch.Tensor.gt_(other)
```

### [paddle.Tensor.greater_than_]()
```python
paddle.Tensor.greater_than_(y)
```

返回 Tensor 的数据类型不一致，PyTorch 返回数据类型与输入 Tensor 一致， Paddle 返回 paddle.bool 类型。
### 参数映射

| PyTorch                          | PaddlePaddle                 | 备注                                                   |
|----------------------------------|------------------------------| ------------------------------------------------------ |
| other  |  y  | 表示输入的 Tensor ，PyTorch 支持 Python Number 和 Tensor 类型， Paddle 仅支持 Tensor 类型。当输入为 Python Number 类型时，需要转写。  |
|  返回值  |  返回值  | 返回 Tensor 的数据类型不一致，PyTorch 返回数据类型与输入 Tensor 一致， Paddle 返回 paddle.bool 类型，需要转写。                                     |
### 转写示例
#### other/返回值：输入为 Number
```python
# PyTorch 写法
result = x.gt_(2)

# Paddle 写法
dtype = x.dtype
result = x.greater_than_(paddle.to_tensor(2)).cast_(dtype)
```
