## torch.asin
### [torch.asin](https://pytorch.org/docs/stable/generated/torch.asin.html?highlight=asin#torch.asin)

```python
torch.asin(input,
           *,
           out=None)
```

### [paddle.asin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/asin_cn.html#asin)

```python
paddle.asin(x,
            name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.asin([-0.5962,  1.4985, -0.4396,  1.4525], out=y)

# Paddle 写法
y = paddle.asin([-0.5962,  1.4985, -0.4396,  1.4525])
```
