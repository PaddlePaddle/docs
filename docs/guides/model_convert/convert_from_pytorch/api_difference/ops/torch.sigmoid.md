## torch.sigmoid
### [torch.sigmoid](https://pytorch.org/docs/stable/generated/torch.sigmoid.html?highlight=sigmoid#torch.sigmoid)

```python
torch.sigmoid(input, *, out=None)
```

### [paddle.nn.functional.sigmoid](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/sigmoid_cn.html#sigmoid)

```python
paddle.nn.functional.sigmoid(x, name=None)
```

功能一致，torch 参数更多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.sigmoid(input)

# Paddle 写法
paddle.nn.functional.sigmoid(x)
```
