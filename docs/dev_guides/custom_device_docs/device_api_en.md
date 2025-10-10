# Device APIs

## initialize 【optional】

### Definition

```c++
C_Status (*initialize)()
```

### Description

It initializes the device backend, such as the runtime or the driver. During the device registration, it is the first to be invoked. But if the API is not implemented, it will not be invoked.

## finalize 【optional】

### Definition

```c++
C_Status (*finalize)()
```

### Description

It deinitializes the device backend. For example, the deinitialization is performed during the exit of the runtime or the driver. The API is invoked till the end of the exit. But if it is not implemented, it will not be invoked.

## init_device 【optional】

### Definition

```c++
C_Status (*init_device)(const C_Device device)
```

### Description

It initializes the designated device and initializes all available devices during the plug-in registration. If not implemented, the API will not be invoked, and it is invoked only after initialization.

### Parameter

device - the device needed to be initialized。

## deinit_device 【optional】

### Definition

```c++
C_Status (*deinit_device)(const C_Device device)
```

### Description

It finalizes the designated device, and deallocate resources allocated to all devices. The API is inovked during the exit. If not implemented, it will not be inovked and it is invoked before finalization.

### Parameter

device - the device needed to be finalized

### Definition

## set_device 【required】

```c++
C_Status (*set_device)(const C_Device device)
```

### Description

It sets the current device, where following tasks are executed.

### Parameter

device - the device needed to be set

## get_device 【required】

### Definition

```c++
C_Status (*get_device)(const C_Device device)
```

### Description

It acquires the current device

### Parameter

device - to store the current device

## synchronize_device 【required】

### Definition

```c++
C_Status (*synchronize_device)(const C_Device device)
```

### Description

It synchronizes the device and waits for the completion of tasks on the device.

### Parameter

device - the device required to be synchronized

## get_device_count 【required】

### Definition

```c++
C_Status (*get_device_count)(size_t* count)
```

### Description

It counts available devices.

### Parameter

count - the number of available devices in storage

## get_device_list 【required】

### Definition

```c++
C_Status (*get_device_list)(size_t* devices)
```

### Description

It acquires the number list of all currently available devices.

### Parameter

devices - numbers of available devices in storage

## get_compute_capability 【required】

### Definition

```c++
C_Status (*get_compute_capability)(size_t* compute_capability)
```

### Description

It gets the computing capability of the device.

### Parameter

compute_capability - the computing capability of the stored device

## get_runtime_version 【required】

### Definition

```c++
C_Status (*get_runtime_version)(size_t* version)
```

### Description

It acquires the runtime version.

### Parameter

version - the runtime version in storage

## get_driver_version 【required】

### Definition

```c++
C_Status (*get_driver_version)(size_t* version)
```

### Description

It gets the driver version.

### Parameter

version - the version of the stored driver

## get_multi_process 【optional】

### Definition

```c++
C_Status (*get_multi_process)(const C_Device device, size_t* multi_process);
```

### Description

Get the number of MultiProcessors on the device.

### Parameter

device - the device to query.
multi_process - to store the number of MultiProcessors.

## get_max_threads_per_mp 【optional】

### Definition

```c++
C_Status (*get_max_threads_per_mp)(const C_Device device, size_t* threads_per_mp);
```

### Description

Get the maximum number of threads per MultiProcessor on the device.

### Parameter

device - the device to query.
threads_per_mp - to store the maximum threads per MultiProcessor.

## get_max_threads_per_block 【optional】

### Definition

```c++
C_Status (*get_max_threads_per_block)(const C_Device device, size_t* threads_per_block);
```

### Description

Get the maximum number of threads per block that can run on the device.

### Parameter

device - the device to query.
threads_per_block - to store the maximum threads per block.

## get_max_grid_dim_size 【optional】

### Definition

```c++
C_Status (*get_max_grid_dim_size)(const C_Device device, std::array<unsigned int, 3>* grid_dim_size);
```

### Description

Get the maximum grid dimension size of the device.

### Parameter

device - the device to query.
grid_dim_size - to store the maximum grid dimension size.

## init_eigen_device 【optional】

### Definition

```c++
C_Status (*init_eigen_device)(C_Place place,
                              C_EigenDevice* eigen_device,
                              C_Stream stream,
                              C_Allocator allocator);
```

### Description

Initialize the Eigen GPU device object.

### Parameter

place - the place object of the device to use.
eigen_device - to store the Eigen GPU device object.
stream - the stream object in Custom Context.
allocator - the allocator object in Custom Context.

## destroy_eigen_device 【optional】

### Definition

```c++
C_Status (*destroy_eigen_device)(const C_Device device,
                                   C_EigenDevice* eigen_device);
```

### Description

Destroy the Eigen GPU device object.

### Parameter

device - the device object to use.
eigen_device - the Eigen GPU device object to be destroyed.
