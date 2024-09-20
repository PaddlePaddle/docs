## [torch 参数更多]torch.concatenate

### [torch.concatenate](https://pytorch.org/docs/stable/generated/torch.concatenate.html#torch-concatenate)

```python
torch.concatenate(tensors, dim=0, out=None)
```

### [paddle.concat](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/concat_cn.html)

```python
paddle.concat(x, axis=0, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射：

| PyTorch | PaddlePaddle | 备注                                                       |
| ------- | ------------ | ---------------------------------------------------------- |
| tensors | x            | 待联结的 list(Tensor) 或者 tuple(Tensor)，仅参数名不一致。 |
| dim     | axis         | 指定对输入进行运算的轴，仅参数名不一致。                   |
| out     | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。               |

### 转写示例：

#### out：输出的 Tensor

```python
# PyTorch 写法:
torch.concatenate(x, y)

# Paddle 写法:
paddle.assign(paddle.concat(x), y)
```
