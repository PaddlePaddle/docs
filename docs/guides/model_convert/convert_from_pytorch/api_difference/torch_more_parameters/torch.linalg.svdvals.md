## [ torch 参数更多 ] torch.linalg.svdvals

### [torch.linalg.svdvals](https://pytorch.org/docs/stable/generated/torch.linalg.svdvals.html#torch.linalg.svdvals)

```python
torch.linalg.svdvals(A, *, driver=None, out=None)
```

### [paddle.linalg.svdvals](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/linalg.py#L3019)

```python
paddle.linalg.svdvals(x, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle  | 备注                                                                                 |
| ------- | ------------- | ------------------------------------------------------------------------------------ |
| A       | x             | 输入 Tensor，仅参数名不一致。                                                        |
| driver  | -             | cuSOLVER 方法名，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。           |
| out     | -             | 表示输出的 Tensor，Paddle 无此参数，需要转写。                                       |

### 转写示例

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.svdvals(A=x, out=y)

# Paddle 写法:
paddle.assign(paddle.linalg.svdvals(x=x), output=y)
```
