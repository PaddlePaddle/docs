## [torch 参数更多]torch.linalg.eigvalsh

### [torch.linalg.eigvalsh](https://pytorch.org/docs/1.13/generated/torch.linalg.eigvalsh.html#torch.linalg.eigvalsh)

```python
torch.linalg.eigvalsh(A, UPLO='L', *, out=None)
```

### [paddle.linalg.eigvalsh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/eigvalsh_cn.html)

```python
paddle.linalg.eigvalsh(x, UPLO='L', name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注 |
| ------- | ------------ | ---------------------------------------------------- |
| A       | x            | 输入 Tensor，仅参数名不一致。                        |
| UPLO    | UPLO         | 表示计算上三角或者下三角矩阵                         |
| out     | -            | 表示输出的 Tensor ， Paddle 无此参数，需要进行转写。 |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.eigvalsh(x, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.eigvalsh(x), y)
```
