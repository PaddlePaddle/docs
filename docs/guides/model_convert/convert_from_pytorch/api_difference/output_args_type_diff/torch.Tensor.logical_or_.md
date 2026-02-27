## [ 返回参数类型不一致 ]torch.Tensor.logical_or_

### [torch.Tensor.logical\_or\_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.logical_or_.html#torch.Tensor.logical_or_)

```python
torch.Tensor.logical_or_(other)
```

### [paddle.Tensor.logical_or_]()

```python
paddle.Tensor.logical_or_(y)
```

返回 Tensor 的数据类型不一致，PyTorch 返回数据类型与输入 Tensor 一致， Paddle 返回 paddle.bool 类型。

### 参数映射

| PyTorch                           | PaddlePaddle                 | 备注                                                   |
|-----------------------------------|------------------------------| ------------------------------------------------------ |
| other  |  y  | 表示输入的 Tensor ，PyTorch 支持 Python Number 和 Tensor 类型， Paddle 仅支持 Tensor 类型。当输入为 Python Number 类型时，需要转写。  |
|  返回值  |  返回值  | 返回 Tensor 的数据类型不一致，PyTorch 返回数据类型与输入 Tensor 一致， Paddle 返回 paddle.bool 类型，需要转写。                                     |


### 转写示例
#### other/返回值
```python
# PyTorch 写法
x.logical_or_(tensor_y)

# Paddle 写法
_x_dtype_ = x.dtype
x.logical_or_(tensor_y).cast(_x_dtype_)
