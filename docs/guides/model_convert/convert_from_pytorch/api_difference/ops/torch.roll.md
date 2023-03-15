## [ 仅参数名不一致 ]torch.roll
### [torch.roll](https://pytorch.org/docs/stable/generated/torch.roll.html?highlight=roll#torch.roll)

```python
torch.roll(input,
           shifts,
           dims=None)
```

### [paddle.roll](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/roll_cn.html#roll)

```python
paddle.roll(x,
            shifts,
            axis=None,
            name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                   |
| shifts         | shifts            | 表示偏移量。                   |
| dims          | axis         | 表示滚动的轴，仅参数名不一致。                          |
