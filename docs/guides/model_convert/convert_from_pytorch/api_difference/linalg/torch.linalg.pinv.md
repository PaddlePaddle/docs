## [torch 参数更多]torch.linalg.pinv

### [torch.linalg.pinv](https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html#torch.linalg.pinv)

```python
torch.linalg.pinv(A, *, atol=None, rtol=None, hermitian=False, out=None)
```

### [paddle.linalg.pinv](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/pinv_cn.html)

```python
paddle.linalg.pinv(x, rcond=1e-15, hermitian=False, name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                               |
| --------- | ------------ | -------------------------------------------------- |
| A         | x            | 输入 Tensor，仅参数名不一致。                      |
| atol      | -            | 绝对阈值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。        |
| rtol      | rcond        | 奇异值（特征值）被截断的阈值，仅参数名不一致。     |
| hermitian | hermitian    | 是否为 hermitian 矩阵或者实对称矩阵。              |
| out       | -            | 表示输出的 Tensor，Paddle 无此参数，需要转写。 |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.pinv(x, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.pinv(x), y)
```
