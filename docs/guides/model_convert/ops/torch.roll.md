## torch.roll
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
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入的 Tensor。                   |
| dims         | axis         | 滚动轴。                          |
