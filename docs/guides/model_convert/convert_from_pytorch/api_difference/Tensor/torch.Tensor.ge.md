## [ 输入参数类型不一致 ]torch.Tensor.ge

### [torch.Tensor.ge](https://pytorch.org/docs/stable/generated/torch.Tensor.ge.html?highlight=torch+tensor+ge#torch.Tensor.ge)

```python
torch.Tensor.ge(other)
```

### [paddle.Tensor.greater_equal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#greater-equal-y-name-none)

```python
paddle.Tensor.greater_equal(y, name=None)
```

其中，PyTorch 与 Paddle 的 `other` 参数所支持类型不一致，具体如下：

### 参数映射

| PyTorch                          | PaddlePaddle                 | 备注                                                   |
|----------------------------------|------------------------------| ------------------------------------------------------ |
| other  |  y  | 输入的 Tensor ，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要转写。 |

### 转写示例
#### other
```python
# PyTorch 写法
result = x.ge(other=2)

# Paddle 写法
result = x.greater_equal(y=paddle.to_tensor(2))
```
