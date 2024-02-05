## [ torch 参数更多 ]torch.mean

### [torch.mean](https://pytorch.org/docs/stable/generated/torch.mean.html)

```python
torch.mean(input, dim, keepdim=False, *, dtype=None, out=None)
```

### [paddle.mean]((url_placeholder))

```python
paddle.mean(x, axis=None, keepdim=False, name=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | -- |
| input   | x            | 输入 Tensor。  |
| dim     | axis         | 指定对 x 进行计算的轴。 |
| keepdim | keepdim      | 是否在输出 Tensor 中保留减小的维度。 |
| dtype   | -            | 输出 Tensor 的类型，Paddle 无此参数, 需要转写。  |
| out     | -            | 输出 Tensor，Paddle 无此参数, 需要转写。   |

### 转写示例

#### dtype：输出数据类型

```python
# PyTorch 写法
torch.mean(x, dtype=torch.float32)

# Paddle 写法
paddle.mean(x).astype(paddle.float32)
```

#### out：输出 Tensor

```python
# PyTorch 写法
torch.mean(x, out=y)

# Paddle 写法
paddle.assign(paddle.mean(x), y)
```
