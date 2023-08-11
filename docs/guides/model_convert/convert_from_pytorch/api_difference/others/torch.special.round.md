## [torch 参数更多]torch.special.round

### [torch.special.round](https://pytorch.org/docs/stable/special.html#torch.special.round)

```python
# torch.special.round 为 torch.round 别名，参数和 torch.round 相同
torch.special.round(input, *, decimals=0, out=None)
```

### [paddle.round](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/round_cn.html)

```python
paddle.round(x, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
| ------- | ------------ | -------------------------------------------------- |
| input   | x            | 输入的 Tensor，仅参数名不一致。                    |
| decimals   | -            | 舍入到的小数位数，PaddlePaddle 无此参数，需要转写。                    |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### decimals：要舍入到的小数位数
```python
# Pytorch 写法
torch.special.round(x, decimals=2)

# Paddle 写法
paddle.round(1e2 * x) / 1e2

# 注：Paddle 可使用 10 的 decimals 次方来实现
```

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.special.round(x, out=y)

# Paddle 写法:
paddle.assign(paddle.round(x), y)
```
