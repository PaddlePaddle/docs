## torch.nn.Unfold
### [torch.nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html?highlight=nn+unfold#torch.nn.Unfold)

```python
torch.nn.Unfold(kernel_size, 
                dilation=1, 
                padding=0, 
                stride=1)
```

### [paddle.nn.Unfold](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Unfold_cn.html#unfold)

```python
paddle.nn.Unfold(kernel_size=[3, 3], 
                    strides=1, 
                    paddings=1, 
                    dilation=1, 
                    name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| padding       | paddings     | 每个维度的扩展，整数或者整型列表。                   |
| stride        | strides      | 卷积步长，整数或者整型列表。                        |  
