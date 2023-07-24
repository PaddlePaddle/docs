## [torch 参数更多]torch.linalg.matmul

### [torch.linalg.matmul](https://pytorch.org/docs/stable/generated/torch.linalg.matmul.html#torch.linalg.matmul)

```python
torch.linalg.matmul(input, other, *, out=None)
```

### [paddle.matmul](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/matmul_cn.html)

```python
paddle.matmul(x, y, transpose_x=False, transpose_y=False, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                      |
| ------- | ------------ | --------------------------------------------------------- |
| input   | x            | 输入变量，仅参数名不一致。                                |
| other   | y            | 输入变量，仅参数名不一致。                                |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。      |
| -       | transpose_x  | 相乘前是否转置 x，PyTorch 无此参数，Paddle 保持默认即可。 |
| -       | transpose_y  | 相乘前是否转置 y，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.matmul(x1, x2, out=y)

# Paddle 写法:
paddle.assign(paddle.matmul(x1, x2), y)
```
