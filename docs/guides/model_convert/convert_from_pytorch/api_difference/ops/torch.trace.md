## [ 仅参数名不一致 ]torch.trace
### [torch.trace](https://pytorch.org/docs/stable/generated/torch.trace.html?highlight=trace#torch.trace)

```python
torch.trace(input)
```
### [paddle.trace](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/trace_cn.html)

```python
paddle.trace(x,
             offset=0,
             axis1=0,
             axis2=1,
             name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。               |
| -             | offset       | 2D-Tensor 中获取对角线的位置，默认值为 0，即主对角线，PyTorch 无此参数，Paddle 保持默认即可。                  |
| -             | axis1        | 当输入的 Tensor 维度大于 2D 时，获取对角线的二维平面的第一维，PyTorch 无此参数，Paddle 保持默认即可。               |
| -             | axis2        | 当输入的 Tensor 维度大于 2D 时，获取对角线的二维平面的第二维，PyTorch 无此参数，Paddle 保持默认即可。                   |
