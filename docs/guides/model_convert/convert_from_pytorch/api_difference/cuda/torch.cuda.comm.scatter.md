## [组合替代实现] torch.cuda.comm.scatter

### [torch.cuda.comm.scatter](https://pytorch.org/docs/stable/generated/torch.cuda.comm.scatter.html)

```python
torch.cuda.comm.scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None, *, out=None)
```

将张量分散到多个设备上，Paddle 无此 API，需要组合替代实现

### 转写示例
```python
# torch 写法
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
torch.cuda.comm.scatter(inputs, devices=devices)

# paddle 写法
def paddle_comm_scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None, out=None):
    if devices is None:
        devices = ['cpu'] * len(tensor)

    if chunk_sizes is not None:
        chunks = paddle.split(tensor, num_or_sections=chunk_sizes, dim=dim)
    else:
        chunks = tensor if isinstance(tensor, list) else [tensor]

    scattered_tensors = out if out is not None else []

    for idx, (chunk, device) in enumerate(zip(chunks, devices)):
        place = paddle.CUDAPlace(int(device.split(':')[-1])) if 'cuda' in device else paddle.CPUPlace()

        tensor_on_device = chunk.cuda(place) if 'cuda' in device else chunk.cpu()

        if streams is not None:
            stream = streams[idx]
            tensor_on_device = tensor_on_device.cuda(place, non_blocking=True)
            tensor_on_device = tensor_on_device.cuda_stream(stream)

        if out is not None:
            out[idx].copy_(tensor_on_device)
        else:
            scattered_tensors.append(tensor_on_device)

    if out is None:
        return scattered_tensors

devices = ['gpu:0', 'gpu:1']
chunk_sizes = [5, 5]
scattered_tensors = paddle_comm_scatter(tensor, devices=devices, chunk_sizes=chunk_sizes)
```
