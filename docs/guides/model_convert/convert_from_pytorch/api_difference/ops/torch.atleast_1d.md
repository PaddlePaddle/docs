## [ 参数不一致 ]torch.atleast_1d

### [torch.atleast_1d](https://pytorch.org/docs/stable/generated/torch.atleast_1d.html#torch-atleast-1d)

```python
torch.atleast_1d(*tensors)
```

### [paddle.atleast_1d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/atleast_1d_cn.html#atleast_1d)

```python
paddle.atleast_1d(*inputs, name=None)
```

PyTorch 与 Paddle 参数不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> tensors </font> | <font color='red'> inputs </font> | 输入的 Tensor ，参数不一致。 |

PyTorch 与 Paddle 功能一致，但对于由多个 Tensor 组成 tuple|list 输入的处理方式略有不同，具体请看转写示例。

### 转写示例

如果有多个 Tensor，如 x 和 y：

```python
# Pytorch 写法
torch.atleast_1d(x, y)

# Paddle 写法
paddle.atleast_1d(x, y)
```

两者功能一致，同为输出两个至少 1 维的 Tensor。

如果 x 和 y 组成 (x, y) 作为输入（注意，此时的输入只有一个，是由 x 和 y 组成的 tuple）：

```python
# Pytorch 写法
torch.atleast_1d((x, y))

# Paddle 写法
paddle.atleast_1d((x, y))
```

PyTorch 仍然输出两个 Tensor，而 Paddle 会将 (x, y) 作为一个整体进行处理并输出。

此时需要注意，x 和 y 需要具有相同的 shape，否则无法将两者转换为一个 Tensor。
