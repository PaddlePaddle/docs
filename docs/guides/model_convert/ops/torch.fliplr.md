## torch.fliplr
### [torch.fliplr](https://pytorch.org/docs/stable/generated/torch.fliplr.html?highlight=fliplr#torch.fliplr)

```python
torch.fliplr(input)
```

### [paddle.flip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flip_cn.html#flip)

```python
paddle.flip(x,
            axis,
            name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                   |
| -             | axis         | 需要翻转的轴，设置 axis = -1 即可。                     |
