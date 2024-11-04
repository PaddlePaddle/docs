## [ torch 参数更多 ]torch.copysign
### [torch.copysign](https://pytorch.org/docs/stable/generated/torch.copysign.html#torch.copysign)

```python
torch.copysign(input,
          other,
          *,
          out=None)
```

### [paddle.copysign](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/copysign_cn.html#copysign)

```python
paddle.copysign(x, y)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input | x |  输入的第一个张量，表示输出的大小，仅参数名不一致。 |
| other  | y            | 输入的第二个张量，表示输出的符号，仅参数名不一致。 |
| out  | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出
```python
# PyTorch 写法
torch.copysign(input, other, out=t)

# Paddle 写法
paddle.assign(paddle.copysign(input, other), y)
```
