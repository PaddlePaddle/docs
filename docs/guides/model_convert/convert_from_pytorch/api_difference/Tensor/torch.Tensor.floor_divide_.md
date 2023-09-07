## [ 参数不一致 ]torch.Tensor.floor_divide_

### [torch.Tensor.floor_divide_](https://pytorch.org/docs/stable/generated/torch.Tensor.floor_divide_.html)

```python
torch.Tensor.floor_divide_(other)
```

### [paddle.Tesnor.floor_divide_]()

```python
paddle.Tensor.floor_divide_(y)
```

其中，PyTorch 与 Paddle 的 `other` 参数所支持类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                        |
| ------- | ------------ | --------------------------- |
| other   | y            | 多维 Tensor，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要转写。 |

### 转写示例
#### other
```python
# PyTorch 写法
result = x.floor_divide_(other=2.)

# Paddle 写法
result = x.floor_divide_(y=paddle.to_tensor(2.))
```
