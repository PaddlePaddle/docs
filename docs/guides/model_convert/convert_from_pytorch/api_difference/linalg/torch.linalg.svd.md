## [ torch 参数更多 ] torch.linalg.svd

### [torch.linalg.svd](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html?highlight=svd#torch.linalg.svd)

```python
torch.linalg.svd(A, full_matrices=True, *, driver=None, out=None)
```

### [paddle.linalg.svd](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/svd_cn.html)

```python
paddle.linalg.svd(x, full_matrices=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle  | 备注                                                                                           |
| ------------- | ------------- | ---------------------------------------------------------------------------------------------- |
| A             | x             | 输入 Tensor，仅参数名不一致。                                                                  |
| full_matrices | full_matrices | 是否计算完整的 U 和 V 矩阵，PyTorch 为 True，Paddle 为 False，Paddle 需设置为与 PyTorch 一致。 |
| driver        | -             | cuSOLVER 方法名，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                     |
| out           | -             | 表示输出的 Tensor，Paddle 无此参数，需要转写。                                                 |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.svd(x, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.svd(x), y)
```
