# Context API

## CustomContext
`CustomContext`为自定义Kernel模板参数Context的实参，具体参照[custom_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/custom/custom_context.h)

```c++
  // 构造函数
  // 参数：place - CustomPlace对象
  // 返回：None
  explicit CustomContext(const CustomPlace&);

  // 析构函数
  virtual ~CustomContext();

  // 获取设备上下文Place信息
  // 参数：None
  // 返回：place - Place对象
  const Place& GetPlace() const override;

  // 获取设备上下文stream信息
  // 参数：None
  // 返回：指针 - stream指针
  void* stream() const;

  // 等待stream上的操作完成
  // 参数：None
  // 返回：None
  void Wait() const override;
```

## DeviceContext
`CustomContext`继承自`DeviceContext`，具体参照[device_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/device_context.h)

```c++
  // 默认构造函数
  DeviceContext();

  // 拷贝构造函数
  DeviceContext(const DeviceContext&);

  // 移动构造函数
  DeviceContext(DeviceContext&&);

  // 移动赋值操作
  DeviceContext& operator=(DeviceContext&&);

  // 默认析构函数
  virtual ~DeviceContext();

  // 设置Device Allocator
  // 参数：Allocator指针
  // 返回：void
  void SetAllocator(const Allocator*);

  // 设置Host Allocator
  // 参数：Allocator指针
  // 返回：void
  void SetHostAllocator(const Allocator*);

  // 设置zero-size Allocator
  // 参数：Allocator指针
  // 返回：void
  void SetZeroAllocator(const Allocator*);

  // 获取 Allocator
  // 参数：None
  // 返回：Allocator对象
  const Allocator& GetAllocator() const;

  // 获取 Host Allocator
  // 参数：None
  // 返回：Allocator对象
  const Allocator& GetHostAllocator() const;

  // 获取 zero-size Allocator
  // 参数：None
  // 返回：Allocator对象
  const Allocator& GetZeroAllocator() const;

  // 为Tensor分配Device内存
  // 参数：TensorBase指针
  //      dtype - DataType类型变量
  //      requested_size - size_t类型变量，默认值为0
  // 返回：void*类型数据指针
  void* Alloc(TensorBase*, DataType dtype, size_t requested_size = 0) const;

  // 为Tensor分配Device内存
  // 模板参数：T - 数据类型
  // 参数：TensorBase指针
  //      requested_size - size_t类型变量，默认值为0
  // 返回：T*类型数据指针
  template <typename T>
  T* Alloc(TensorBase* tensor, size_t requested_size = 0) const;

  // 为Tensor分配Host内存
  // 参数：TensorBase指针
  //      dtype - DataType类型变量
  //      requested_size - size_t类型变量，默认值为0
  // 返回：void*类型数据指针
  void* HostAlloc(TensorBase* tensor,
                  DataType dtype,
                  size_t requested_size = 0) const;

  // 为Tensor分配Host内存
  // 模板参数：T - 数据类型
  // 参数：TensorBase指针
  //      requested_size - size_t类型变量，默认值为0
  // 返回：T*类型数据指针
  template <typename T>
  T* HostAlloc(TensorBase* tensor, size_t requested_size = 0) const;

  // 获取设备上下文Place信息，需子类实现
  // 参数：None
  // 返回：place - Place对象
  virtual const Place& GetPlace() const = 0;

  // 等待stream上的操作完成，需子类实现
  // 参数：None
  // 返回：None
  virtual void Wait() const {}

  // 为特定算子设置Generator
  // 参数：Generator*
  // 返回：None
  void SetGenerator(Generator*);

  // 获取Generator
  // 参数：None
  // 返回：Generator*
  Generator* GetGenerator() const;

  // 为特定算子设置Host Generator
  // 参数：Generator*
  // 返回：None
  void SetHostGenerator(Generator*);

  // 获取Host Generator
  // 参数：None
  // 返回：Generator*
  Generator* GetHostGenerator() const;

```
