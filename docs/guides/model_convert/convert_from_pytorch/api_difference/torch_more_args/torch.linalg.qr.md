## [ torch 参数更多 ]torch.linalg.qr
### [torch.linalg.qr](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html#torch.linalg.qr)
```python
torch.linalg.qr(A, mode='reduced', *, out=None)
```

### [paddle.linalg.qr](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/qr_cn.html#paddle.linalg.qr)
```python
paddle.linalg.qr(x, mode='reduced', name=None, *, out=None)
```

两者功能一致，参数名和返回值有差异，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                               |
| ------- | ------------ | -------------------------------------------------- |
| A       | x            | 输入 Tensor，仅参数名不一致。                      |
| mode    | mode         | 控制正交三角分解的行为。                           |
| out     | out          | 输出的 Tensor 元组，Paddle 已支持此参数。 |

### 返回值差异

`mode='r'` 时，两者返回值不同：
- **PyTorch**：返回 `(Q, R)` 具名元组，Q 为空张量。
- **Paddle**：直接返回单个 Tensor **R**（不包含 Q）。

因此，对 ``mode='r'`` 的返回值需注意解包方式。

### 转写示例
#### mode='r' 的返回值
```python
# PyTorch 写法:
r = torch.linalg.qr(x, mode='r')  # 返回 (Q, R) 元组, R 在索引 1

# Paddle 写法:
r = paddle.linalg.qr(x, mode='r')  # 直接返回 Tensor R
```

#### out 参数：输出的 Tensor
```python
# PyTorch 写法 (mode='reduced'):
torch.linalg.qr(x, out=(q, r))

# Paddle 写法 (mode='reduced'):
paddle.linalg.qr(x, out=(q, r))

# PyTorch 写法 (mode='r'，out 为 (Q, R) 元组):
torch.linalg.qr(x, mode='r', out=(q, r))

# Paddle 写法 (mode='r'，out 为单个 Tensor):
paddle.linalg.qr(x, mode='r', out=r)
```
