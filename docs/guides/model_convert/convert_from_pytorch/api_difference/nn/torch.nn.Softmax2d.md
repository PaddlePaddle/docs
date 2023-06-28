## [用法不同]torch.nn.Softmax2d

### [torch.nn.Softmax2d](https://pytorch.org/docs/stable/generated/torch.nn.Softmax2d.html?highlight=softmax2d#torch.nn.Softmax2d)

```
torch.nn.Softmax2d(*args,
           **kwargs)
```

### [paddle.nn.Softmax](no)

```
paddle.nn.Softmax(axis=-1)
```

其中 Paddle 并没有 torch.nn.Softmax2d 此 api ，可通过 paddle.nn.Softmax 设置参数 axis 为 -3 实现同样的效果：

### 转写示例

```
# Pytorch 写法
cri = torch.nn.Softmax2d()
cri(input)

# Paddle 写法
cri = paddle.nn.Softmax(axis=-3)
cri(input)
```
