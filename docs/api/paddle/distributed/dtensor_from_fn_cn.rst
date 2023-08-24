.. _cn_api_distributed_dtensor_from_fn:

dtensor_from_fn
-------------------------------

.. py:class:: paddle.distributed.dtensor_from_fn(fn,dist_attr,*args,**kwargs)

通过传入的函数fn以及可能的参数*args,**kwargs构造一个tensor，将这个tensor传给shard_tensor


参数
:::::::::

    - **fn**  - 类似empty/ones/zeros等任意函数
    - **dist_attr** (paddle.distributed.DistAttr) - 描述 Tensor 在 ProcessMesh 上的分布或切片方式。
    - ***args**  - fn函数可能存在的参数
    - ****kwargs**  - fn函数可能存在的参数
    

返回
:::::::::
带有分布式信息的 Tensor



**代码示例**

COPY-FROM: paddle.distributed.dtensor_from_fn
