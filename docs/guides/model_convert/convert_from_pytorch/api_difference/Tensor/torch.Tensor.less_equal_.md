## [ 输入参数类型不一致 ]torch.Tensor.less_equal_

### [torch.Tensor.less_equal_](https://pytorch.org/docs/stable/generated/torch.Tensor.less_equal_.html)

```python
torch.Tensor.less_equal_(other)
```

### [paddle.Tensor.less_equal_]()

```python
paddle.Tensor.less_equal_(y)
```

其中 Paddle 和 PyTorch 的 `other` 参数所支持的数据类型不一致，具体如下：
### 参数映射

| PyTorch                          | PaddlePaddle                 | 备注                                                   |
|----------------------------------|------------------------------| ------------------------------------------------------ |
| other  |  y  | 表示输入的 Tensor ，PyTorch 支持 Python Number 和 Tensor 类型， Paddle 仅支持 Tensor 类型。当输入为 Python Number 类型时，需要转写。  |

### 转写示例
#### other：输入为 Number
```python
# PyTorch 写法
result = x.less_equal_(2)

# Paddle 写法
result = x.less_equal_(paddle.to_tensor(2))
```
