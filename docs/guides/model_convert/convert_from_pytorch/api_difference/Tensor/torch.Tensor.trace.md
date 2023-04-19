## [仅 paddle 参数更多]troch.Tensor.trace

### [torch.Tensor.trace](https://pytorch.org/docs/1.13/generated/torch.Tensor.trace.html#torch-tensor-trace)

```python
torch.Tensor.trace()
```

### [paddle.Tensor.trace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#trace-offset-0-axis1-0-axis2-1-name-none)

```python
paddle.Tensor.trace(offset=0, axis1=0, axis2=1, name=None)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle |                             备注                             |
| :-----: | :----------: | :----------------------------------------------------------: |
|    -    |    offset    | 表示指定的二维平面中获取对角线的位置，PyTorch 无此参数，Paddle 保持默认即可。 |
|    -    |    axis1     | 表示获取对角线的二维平面的第一维，PyTorch 无此参数，Paddle 保持默认即可。 |
|    -    |    axis2     | 表示获取对角线的二维平面的第二维，PyTorch 无此参数，Paddle 保持默认即可。 |
