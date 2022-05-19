# Context APIs

## CustomContext
`CustomContext` is the acutal parameter of the template parameter Context of the custom kernel function. For details, please refer to [custom_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/custom/custom_context.h).

```c++
  // Constructor
  // Parameter：place - CustomPlace object
  // Return：None
  explicit CustomContext(const CustomPlace&);

  // Destructor
  virtual ~CustomContext();

  // Get the contextual place in the device
  // Parameter：None
  // Return：place - Place object
  const Place& GetPlace() const override;

  // Get the contextual stream in the device
  // Parameter：None
  // Return：stream - void* pointer
  void* stream() const;

  // Wait for the completion of operations on the stream
  // Parameter：None
  // Return：None
  void Wait() const override;
```

## DeviceContext
`CustomContext` originates from `DeviceContextp`,please refer to [device_context.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/device_context.h)

```c++
  // No-Parameter constructor
  DeviceContext();

  // Copy constructor
  DeviceContext(const DeviceContext&);

  // Move constructor
  DeviceContext(DeviceContext&&);

  // Move assignment operator
  DeviceContext& operator=(DeviceContext&&);

  // Destructor
  virtual ~DeviceContext();

  // Set device allocator
  // Parameter：Allocator pointer
  // Return：None
  void SetAllocator(const Allocator*);

  // Set host allocator
  // Parameter：Allocator pointer
  // Return：None
  void SetHostAllocator(const Allocator*);

  // Set zero-size allocator
  // Parameter：Allocator pointer
  // Return：None
  void SetZeroAllocator(const Allocator*);

  // Get Allocator
  // Parameter：None
  // Return：Allocator object
  const Allocator& GetAllocator() const;

  // Get Host allocator
  // Parameter：None
  // Return：Allocator object
  const Allocator& GetHostAllocator() const;

  // Get zero-size allocator
  // Parameter：None
  // Return：Allocator object
  const Allocator& GetZeroAllocator() const;

  // Allocate the device memory for Tensor
  // Parameter: TensorBase pointer
  //      dtype - DataType variable
  //      requested_size - size_t variable with the default value of 0
  // Return：data pointer - void* pointer
  void* Alloc(TensorBase*, DataType dtype, size_t requested_size = 0) const;

  // Allocate device memory for Tensor
  // Template Parameter：T - data type
  // Parameter：TensorBase pointer
  //      requested_size - size_t variable, 0 by default
  // Return：data pointer - T* pointer
  template <typename T>
  T* Alloc(TensorBase* tensor, size_t requested_size = 0) const;

  // Allocate host memory for Tensor
  // Parameter：TensorBase pointer
  //      dtype - DataType variable
  //      requested_size - size_t variable, 0 by default
  // Return：data pointer - void* pointer
  void* HostAlloc(TensorBase* tensor,
                  DataType dtype,
                  size_t requested_size = 0) const;

  // Allocate host storage for Tensor
  // Template Parameter：T - data type
  // Parameter：TensorBase pointer
  //      requested_size - size_t variable, 0 by default
  // Return：data pointer - T* data pointer
  template <typename T>
  T* HostAlloc(TensorBase* tensor, size_t requested_size = 0) const;

  // Get the contextual information of the place, and implement sub interfaces
  // Parameter：None
  // Return：place - Place object
  virtual const Place& GetPlace() const = 0;

  // Wait for the completion of operations on the stream, and implement sub interfaces
  // Parameter：None
  // Return：None
  virtual void Wait() const {}

  // Set the random number generator
  // Parameter：Generator pointer
  // Return：None
  void SetGenerator(Generator*);

  // Get the random number generator
  // Parameter：None
  // Return：Generator pointer
  Generator* GetGenerator() const;

  // Set the Host random number generator
  // Parameter：Generator pointer
  // Return：None
  void SetHostGenerator(Generator*);

  // Get the Host random number generator
  // Parameter：None
  // Return：Generator pointer
  Generator* GetHostGenerator() const;

```

## Relevant Information

- `Place` and `CustomPlace`：please refer to [place.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/place.h)
- `Allocation` and `Allocator`：please refer to [allocator.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/allocator.h)
- `TensorBase`：please refer to [tensor_base.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/tensor_base.h)
- `DataType`：please refer to [data_type.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/data_type.h)
- `Generator`：please refer to [generator.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/generator.h)
