## [ torch 参数更多 ] torch.linalg.svdvals

### [torch.linalg.svdvals](https://pytorch.org/docs/stable/generated/torch.linalg.svdvals.html#torch.linalg.svdvals)

```python
torch.linalg.svdvals(A, *, driver=None, out=None)
```

### [paddle.linalg.svd](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/svd_cn.html)

```python
paddle.linalg.svd(x, full_matrices=False, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle  | 备注                                                                                 |
| ------- | ------------- | ------------------------------------------------------------------------------------ |
| A       | x             | 输入 Tensor，仅参数名不一致。                                                        |
| driver  | -             | cuSOLVER 方法名，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。           |
| -       | full_matrices | 是否计算完整的 U 和 V 矩阵，Paddle 为 False，PyTorch 无此参数，Paddle 使用默认即可。 |
| out     | -             | 表示输出的 Tensor，Paddle 无此参数，需要转写。                                       |
| 返回值  | 返回值        | PyTorch 返回值为 S，Paddle 返回 U、S、VH，需要转写。                                 |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.svdvals(x, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.svd(x)[1], y)
```

#### 返回值

```python
# PyTorch 写法:
y = torch.linalg.svdvals(x)

# Paddle 写法:
y = paddle.linalg.svd(x)[1]
```
