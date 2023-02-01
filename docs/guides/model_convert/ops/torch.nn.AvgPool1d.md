## torch.nn.AvgPool1d
### [torch.nn.AvgPool1d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html?highlight=nn+avgpool1d#torch.nn.AvgPool1d)

```python
torch.nn.AvgPool1d(kernel_size, 
                    stride=None, 
                    padding=0, 
                    ceil_mode=False, 
                    count_include_pad=True)
```

### [paddle.nn.AvgPool1D](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/AvgPool1D_cn.html#avgpool1d)

```python
paddle.nn.AvgPool1D(kernel_size, 
                    stride=None, 
                    padding=0, 
                    exclusive=True, 
                    ceil_mode=False, 
                    name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| count_include_pad| exclusive | 是否用额外 padding 的值计算平均池化结果。                   |
