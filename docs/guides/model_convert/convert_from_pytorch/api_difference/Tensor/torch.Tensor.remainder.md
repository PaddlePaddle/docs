## [ 参数不一致 ]torch.Tensor.remainder
### [torch.Tensor.remainder](https://pytorch.org/docs/stable/generated/torch.Tensor.remainder.html?highlight=torch+tensor+remainder#torch.Tensor.remainder)

```python
torch.Tensor.remainder(divisor)
```

### [paddle.Tensor.remainder](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#remainder-y-name-none)

```python
paddle.Tensor.remainder(y, name=None)
```

其中 Paddle 与 PyTorch 运算除数参数所支持的类型不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| divisor         | y            | 除数，PyTorch 可为 Tensor or Scalar，Paddle 仅可为 Tensor, 需要转写。   |

### 转写示例
#### divisor：除数为 Scalar
```python
# PyTorch 写法
x.remainder(1)

# Paddle 写法
x.remainder(y=paddle.to_tensor(1))
```
