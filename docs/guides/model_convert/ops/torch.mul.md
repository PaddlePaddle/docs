## torch.mul
### [torch.mul](https://pytorch.org/docs/stable/generated/torch.mul.html?highlight=mul#torch.mul)

```python
torch.mul(input,
          other,
          *,
          out=None)
```

### [paddle.multiply](https://vpaddlepaddle.org.cn/documentation/docs/zh/api/paddle/multiply_cn.html#multiply)

```python
paddle.multiply(x,
                y,
                name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| other         | y            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.mul([[3, 5]], [[1], [2]], out=y)

# Paddle 写法
y = paddle.multiply([[3, 5]], [[1], [2]])
```
