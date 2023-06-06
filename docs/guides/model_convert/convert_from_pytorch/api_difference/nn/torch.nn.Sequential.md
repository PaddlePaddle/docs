## [ 参数不一致 ]torch.nn.Sequential
### [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#sequential)

```python
torch.nn.Sequential(args)
```

### [paddle.nn.Sequential](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Sequential_cn.html#sequential)

```python
paddle.nn.Sequential(layers)
```

其中 Paddle 与 Pytorch 的 输入所支持的参数类型不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| args       | layers     |  序列化容器传入参数，paddle 传入 Layers 或可迭代的 name Layer 对，torch 支持 Module 类型和 OrderedDict 类型，类型不一致，需要进行转写。  |

### 转写示例
#### *arg
```python
# Pytorch 写法
torch.nn.Sequential(torch.nn.Relu())

# Paddle 写法
paddle.nn.Sequential(paddle.nn.Relu())
```
#### arg: OrderedDict[str, Module]
```python
# Pytorch 写法
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
        ]))

# Paddle 写法
model = paddle.nn.Sequential(*[('conv1', paddle.nn.Conv2D(in_channels=1,
    out_channels=20, kernel_size=5)), ('relu1', paddle.nn.ReLU())])
```
