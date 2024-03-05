## [ torch 参数更多 ]torch.nanquantile

### [torch.nanquantile](https://pytorch.org/docs/stable/generated/torch.nanquantile.html?highlight=nanquantile#torch.nanquantile)

```python
torch.nanquantile(input,
               q,
               dim=None,
               keepdim=False,
               *,
               interpolation='linear',
               out=None)
```

### [paddle.nanquantile]()

```python
paddle.nanquantile(x,
                q,
                axis=None,
                keepdim=False,
                name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input |  x  | 表示输入的 Tensor，仅参数名不一致。  |
|   q   |  q  | 待计算的分位数。  |
|  dim  | axis| 指定对 x 进行计算的轴，仅参数名不一致。 |
|keepdim|keepdim| 是否在输出 Tensor 中保留减小的维度。|
|interpolation|  - | 当所需分位数位于两个数据点之间时使用的插值方法，Paddle 无此参数，Paddle 暂无转写方式。|
|  out  |  -  | 表示输出的 Tensor，Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.nanquantile(torch.tensor([float('nan'), 1., 2., 3.]), 0.6, out=y)

# Paddle 写法
paddle.assign(paddle.nanquantile(paddle.to_tensor([float('nan'), 1., 2., 3.]), 0.6), y)
```
