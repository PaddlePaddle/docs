## [ torch 参数更多 ]torch.hypot

### [torch.hypot](https://pytorch.org/docs/stable/generated/torch.hypot.html#torch.hypot)

```python
torch.hypot(input, other, *, out=None)
```

### [paddle.hypot](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/hypot_cn.html)

```python
paddle.hypot(x, y, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                           |
| ------- | ------------ | ---------------------------------------------- |
| input   | x            | 输入 Tensor，仅参数名不一致。                  |
| other   | y            | 输入 Tensor，仅参数名不一致。                  |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out：指定输出

```python
# PyTorch 写法
torch.hypot(x, y, out=out)

# Paddle 写法
paddle.assign(paddle.hypot(x, y), out)
```
