# Tensor API

## Tensor信息获取：
    - `int64_t numel() const` 返回Tensor元素数量
    - `const DDim& dims() const` 返回Tensor的dims信息
    - `DataType dtype() const` 返回Tensor元素的数据类型
    - `DataLayout layout() const` 返回Tensor的layout信息
    - `const Place& place() const` 返回Tensor的place

## Tensor操作：
    - `void set_meta(DenseTensorMeta&& meta)` 设置Tensor的Meta信息
    - `DenseTensor& Resize(const DDim& dims)` 修改Tensor的dims
    - `DenseTensor& ShareDataWith(const DenseTensor& src)`两个Tensor共享相同内存
