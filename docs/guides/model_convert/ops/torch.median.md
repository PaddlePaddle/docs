## torch.median
### [torch.median](https://pytorch.org/docs/stable/generated/torch.median.html?highlight=median#torch.median)

```python
torch.median(input,
             dim=- 1,
             keepdim=False,
             *,
             out=None)
```

### [paddle.median](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/median_cn.html#median)

```python
paddle.median(x,
              axis=None,
              keepdim=False,
              name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                   |
| dim           | axis         | 指定对 x 进行计算的轴。                   |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.median([3, 5], dim=0, out=y)

# Paddle 写法
y = paddle.median([3, 5], axis=0)
```
