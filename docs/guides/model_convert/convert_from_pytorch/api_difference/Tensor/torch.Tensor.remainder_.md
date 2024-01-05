## [ 参数不一致 ]torch.Tensor.remainder_
### [torch.Tensor.remainder_](https://pytorch.org/docs/stable/generated/torch.Tensor.remainder_.html?highlight=torch+tensor+remainder_#torch.Tensor.remainder_)

```python
torch.Tensor.remainder_(divisor)
```

### [paddle.remainder_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/remainder__cn.html#remainder)

```python
paddle.remainder_(x, y)
```


其中 Paddle 与 PyTorch 运算除数参数所支持的类型不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| divisor         | y            | 除数，PyTorch 可为 Tensor or Scalar，Paddle 仅可为 Tensor，需要转写。   |

### 转写示例
#### divisor：除数为 Scalar
```python
# PyTorch 写法
x.remainder_(1)

# Paddle 写法
paddle.remainder_(x, y=paddle.to_tensor(1))
```
