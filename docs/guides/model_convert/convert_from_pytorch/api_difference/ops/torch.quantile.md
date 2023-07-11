## [ torch 参数更多 ]torch.quantile

### [torch.quantile](https://pytorch.org/docs/2.0/generated/torch.quantile.html?highlight=quantile#torch.quantile)

```python
torch.quantile(input,
               q,
               dim=None,
               keepdim=False,
               *,
               interpolation='linear',
               out=None)
```

### [paddle.quantile](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/quantile_cn.html)

```python
paddle.quantile(x,
                q,
                axis=None,
                keepdim=False,
                name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input |  x  | 表示输入的 Tensor，仅参数名不一致。  |
|   q   |  q  | 待计算的分位数。  |
|  dim  | axis| 指定对 x 进行计算的轴，仅参数名不一致。 |
|keepdim|keepdim| 是否在输出 Tensor 中保留减小的维度。|
|interpolation|  - | 当所需分位数位于两个数据点之间时使用的插值方法，Paddle 无此参数，需要进行转写。|
|  out  |  -  | 表示输出的 Tensor，Paddle 无此参数，需要进行转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.quantile(torch.tensor([0., 1., 2., 3.]), 0.6, interpolation='linear', out=y)

# Paddle 写法
paddle.assign(paddle.quantile(paddle.to_tensor([0., 1., 2., 3.]), 0.6), y)
```
