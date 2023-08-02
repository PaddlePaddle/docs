## [torch 参数更多 ]torch.remainder
### [torch.remainder](https://pytorch.org/docs/stable/generated/torch.remainder.html?highlight=remainder#torch.remainder)

```python
torch.remainder(input,
                other,
                *,
                out=None)
```

### [paddle.remainder](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/remainder_cn.html#remainder)

```python
paddle.remainder(x,
                 y,
                 name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 被除数，Pytorch 可为 Tensor or Scalar，Paddle 仅可为 Tensor。  |
| other         | y            | 除数，Pytorch 可为 Tensor or Scalar，Paddle 仅可为 Tensor。   |
| out           | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.remainder([3, 5], 2, out=y)

# Paddle 写法
paddle.assign(paddle.remainder([3, 5], 2), y)
```
