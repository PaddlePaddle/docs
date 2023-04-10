## [ 参数不一致 ]torch.utils.data.DataLoader
### [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)
```python
torch.utils.data.DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            sampler=None,
                            batch_sampler=None,
                            num_workers=0,
                            collate_fn=None,
                            pin_memory=False,
                            drop_last=False,
                            timeout=0,
                            worker_init_fn=None,
                            multiprocessing_context=None,
                            generator=None,
                            *,
                            prefetch_factor=2,
                            persistent_workers=False,
                            pin_memory_device='')
```

### [paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)
```python
paddle.io.DataLoader(dataset,
                     feed_list=None,
                     places=None,
                     return_list=False,
                     batch_sampler=None,
                     batch_size=1,
                     shuffle=False,
                     drop_last=False,
                     collate_fn=None,
                     num_workers=0,
                     use_buffer_reader=True,
                     use_shared_memory=False,
                     timeout=0,
                     worker_init_fn=None)
```


### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| sampler       | -            | 表示数据集采集器，PaddlePaddle 无此参数。  |
| pin_memory    | -            | 表示数据最开始是属于锁页内存，PaddlePaddle 无此参数。 |
| multiprocessing_context | -  | ？，PaddlePaddle 无此参数。                   |
| generator     | -            | 用于采样的伪随机数生成器，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| prefetch_factor | -          | 表示每个 worker 预先加载的数据数量，PaddlePaddle 无此参数。  |
| persistent_workers | -       | 表示数据集使用一次后，数据加载器将会不会关闭工作进程，PaddlePaddle 无此参数。  |
| pin_memory_device  | -       | 数据加载器是否在返回 Tensor 之前将 Tensor 复制到设备固定存储器中，PaddlePaddle 无此参数。  |
| -             | feed_list    | 表示 feed 变量列表，PyTorch 无此参数。                   |
| -             | places       | 数据需要放置到的 Place 列表，PyTorch 无此参数。                   |
| -             | return_list  | 每个设备上的数据是否以 list 形式返回，PyTorch 无此参数。                   |
| -             | use_buffer_reader | 表示是否使用缓存读取器，PyTorch 无此参数。                   |
| -             | use_shared_memory | 表示是否使用共享内存来提升子进程将数据放入进程间队列的速度，PyTorch 无此参数。   |

### 功能差异
#### 自定义数据采集器
***PyTorch***：可通过设置`sampler`自定义数据采集器。
***PaddlePaddle***：PaddlePaddle 无此功能，可使用如下代码自定义一个 DataLoader 实现该功能。
```python
class DataLoader(paddle.io.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False

        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        if sampler is not None:
            self.batch_sampler.sampler = sampler
```
