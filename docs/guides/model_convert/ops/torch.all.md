## torch.all
### [torch.all](https://pytorch.org/docs/stable/generated/torch.all.html?highlight=all#torch.all)

```python
torch.all(input)
```

### [paddle.all](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/all_cn.html#all)

```python
paddle.all(x,
           axis=None,
           keepdim=False,
           name=None)
```

其中 Paddle 相比 Pytorch 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的多维 Tensor。                   |
| -             | axis         | 计算逻辑与运算的维度，Pytorch 无，保持默认即可。               |
| -             | keepdim      | 是否在输出 Tensor 中保留减小的维度，Pytorch 无，保持默认即可。  |
