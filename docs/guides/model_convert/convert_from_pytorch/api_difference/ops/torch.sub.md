## [ torch 参数更多 ]torch.sub
### [torch.sub](https://pytorch.org/docs/stable/generated/torch.sub.html?highlight=torch%20sub#torch.sub)

```python
torch.sub(input, other, *, alpha=1, out=None)
```

### [paddle.subtract](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/subtract_cn.html#subtract)

```python
paddle.subtract(x, y, name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| alpha         | -            | 表示`other`的乘数，PaddlePaddle 无此参数。  |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。  |


### 功能差异

#### 计算差异
***PyTorch***：
$ out = input - alpha * other $

***PaddlePaddle***：
$ out = x - y $
