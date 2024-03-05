## [仅 paddle 参数更多]torch.linalg.vander

### [torch.linalg.vander](https://pytorch.org/docs/stable/generated/torch.linalg.vander.html#torch.linalg.vander)

```python
torch.linalg.vander(x, N=None)
```

### [paddle.vander](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vander_cn.html)

```python
paddle.vander(x, n=None, increasing=False, name=None)
```

其中 Paddle 相比 PyTorch 支持更多参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                               |
| ------- | ------------ | ------------------------------------------------------------------ |
| x       | x            | 输入的 Tensor。                                                    |
| N       | n            | 输出中的列数, 仅参数名不一致。                                     |
| -       | increasing   | 列的幂次顺序，PyTorch 无此参数，Paddle 设置为 True，需要转写。 |

### 转写示例

#### increasing：列的幂次顺序

```python
# PyTorch 写法
torch.linalg.vander(x)

# Paddle 写法
paddle.vander(x, increasing=True)
```
