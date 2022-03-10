# Context API

## 设备信息获取API：
    - `void* stream() const` 返回void*类型的`stream`
    - `const Place& GetPlace()` 返回当前设备的`Place`

## 设备内存分配API：
    - `template <typename T> T* Alloc(TensorBase* tensor, size_t requested_size = 0) const` 为给定的tensor指针分配内存
