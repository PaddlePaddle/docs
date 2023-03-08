## torch.nonzero
### [torch.nonzero](https://pytorch.org/docs/1.13/generated/torch.nonzero.html?highlight=nonzero#torch.nonzero)

```python
torch.nonzero(input,
              *,
              out=None,
              as_tuple=False)
```

### [paddle.nonzero](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nonzero_cn.html#nonzero)

```python
paddle.nonzero(x,
               as_tuple=False)
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
torch.nonzero(x, out=y)

# Paddle 写法
y = paddle.nonzero(x)
```
