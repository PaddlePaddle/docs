## [ 参数不一致 ]torch.Tensor.greater_equal

### [torch.Tensor.greater_equal](https://pytorch.org/docs/1.13/generated/torch.Tensor.greater_equal.html?highlight=torch+tensor+greater_equal#torch.Tensor.greater_equal)

```python
torch.Tensor.greater_equal(other)
```

### [paddle.Tensor.greater_equal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#greater-equal-y-name-none)

```python
paddle.Tensor.greater_equal(y, name=None)
```

其中，PyTorch 与 Paddle 的 `other` 参数所支持类型不一致，具体如下：

### 参数映射
| PyTorch                          | PaddlePaddle                 | 备注                                                   |
|----------------------------------|------------------------------| ------------------------------------------------------ |
| other  |  y  | 输入的 Tensor ，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要进行转写。 |

### 转写示例
#### other
```python
# PyTorch 写法
result = x.greater_equal(other=2)

# Paddle 写法
result = x.greater_equal(y=paddle.to_tensor(2))
```
