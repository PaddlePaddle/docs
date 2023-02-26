## torch.erf
### [torch.erf](https://pytorch.org/docs/stable/generated/torch.erf.html?highlight=erf#torch.erf)

```python
torch.erf(input,
          *,
          out=None)
```

### [paddle.erf](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/erf_cn.html#erf)

```python
paddle.erf(x,
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
torch.erf([0, -1., 10.], out=y)

# Paddle 写法
y = paddle.erf([0, -1., 10.])
```
