## torch.cosh
### [torch.cosh](https://pytorch.org/docs/stable/generated/torch.cosh.html?highlight=cosh#torch.cosh)

```python
torch.cosh(input,
            *,
            out=None)
```

### [paddle.cosh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cosh_cn.html#cosh)

```python
paddle.cosh(x,
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
torch.cosh([0.1632,  1.1835, -0.6979, -0.7325], out=y)

# Paddle 写法
y = paddle.cosh([0.1632,  1.1835, -0.6979, -0.7325])
```
