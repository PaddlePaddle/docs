## [ 仅参数名不一致 ]torch.tile
### [torch.tile](https://pytorch.org/docs/stable/generated/torch.tile.html?highlight=tile#torch.tile)

```python
torch.tile(input,
           dims)
```

### [paddle.tile](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tile_cn.html#tile)

```python
paddle.tile(x,
            repeat_times,
            name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                   |
| dims          | repeat_times | 指定输入 x 每个维度的复制次数，仅参数名不一致。 |
