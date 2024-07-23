## [torch 参数更多]torch.concat

### [torch.concat](https://pytorch.org/docs/stable/generated/torch.concat.html#torch.concat)

```python
torch.concat(tensors, dim=0, *, out=None)
```

### [paddle.concat](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/concat_cn.html)

```python
paddle.concat(x, axis=0, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                     |
| ------- | ------------ | -------------------------------------------------------- |
| tensors | x            | 待联结的 Tensor list 或者 Tensor tuple，仅参数名不一致。 |
| dim     | axis         | 指定对输入 x 进行运算的轴，仅参数名不一致。              |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。       |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.concat(x, out=y)

# Paddle 写法:
paddle.assign(paddle.concat(x), y)
```
