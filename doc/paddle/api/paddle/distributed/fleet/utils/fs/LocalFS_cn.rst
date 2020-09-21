.. _cn_api_distributed_fleet_utils_fs_LocalFS:

LocalFS
-------------------------------

.. py:class:: paddle.distributed.fleet.utils.LocalFS
一个本地文件系统工具类。

**示例代码**：
.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    subdirs, files = client.ls_dir("./")

.. py:method:: ls_dir(fs_path)
列出 `fs_path` 路径下所有的文件和子目录。

参数：
    - **fs_path** (str): 本地文件路径。

返回：
    - Tuple， 一个包含所有子目录和文件名的2-Tuple，格式形如: ([subdirname1, subdirname1, ...], [filename1, filename2, ...])。

**示例代码**：

.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    subdirs, files = client.ls_dir("./")

.. py:method:: mkdirs(fs_path)
创建一个本地目录。

参数：
    - **fs_path** (str): 本地文件路径。

**示例代码**：

.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    client.mkdirs("test_mkdirs")
    client.delete("test_mkdirs")

.. py:method:: rename(fs_src_path, fs_dst_path)
重命名本地文件名。

参数：
    - **fs_src_path** (str)：重命名前原始文件名。
    - **fs_dst_path** (str)：新文件名。

**示例代码**：

.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    client.touch("test_rename_src")
    print(client.is_exists("test_rename_src")) # True
    client.rename("test_rename_src", "test_rename_dst")
    print(client.is_exists("test_rename_src")) # False
    print(client.is_exists("test_rename_dst")) # True
    client.delete("test_rename_dst")

.. py:method:: delete(fs_path)
删除本地文件（或目录）。

参数：
    - **fs_path** (str): 本地文件路径。

**示例代码**：

.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    client.mkdirs("test_localFS_mkdirs")
    client.delete("test_localFS_mkdirs")

.. py:method:: is_file(fs_path)
判断当前路径是否是一个文件。

参数：
    - **fs_path** (str): 本地文件路径。

返回：
    - Bool：若当前路径存在且是一个文件，返回 `True` ，反之则返回 `False` 。

**示例代码**：

.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    client.touch("test_is_file")
    print(client.is_file("test_is_file")) # True
    client.delete("test_is_file")

.. py:method:: is_dir(fs_path)
判断当前路径是否是一个目录。

参数：
    - **fs_path** (str): 本地文件路径。

返回：
    - Bool：若当前路径存在且是一个目录，返回 `True` ，反之则返回 `False` 。

**示例代码**：

.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    client.mkdirs("test_is_dir")
    print(client.is_dir("test_is_file")) # True
    client.delete("test_is_dir")

.. py:method:: is_exist(fs_path)
判断当前路径是否存在。

参数：
    - **fs_path** (str): 本地文件路径。

返回：
    - Bool：若当前路径存在返回 `True` ，反之则返回 `False` 。

**示例代码**：

.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    ret = local_fs.is_exist("test_is_exist")

.. py:method:: touch(fs_path, exist_ok=True)
创建一个本地文件。

参数：
    - **fs_path** (str): 本地文件路径。
    - **exist_ok** (bool): 文件路径已存在时程序是否报错。若 `exist_ok = True`，则直接返回，反之则抛出文件存在的异常，默认不抛出异常。

**示例代码**：

.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    client.touch("test_touch")
    client.delete("test_touch")

.. py:method:: mv(src_path, dst_path, overwrite=False)
本地文件移动。

参数：
    - **src_path** (str): 移动前源文件路径名。
    - **dst_path** (str): 移动后目标文件路径名。
    - **overwrite** (bool): 若目标文件已存在，是否删除进行重写，默认不重写并抛出异常。
 
**示例代码**：

.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    client.touch("test_mv_src")
    client.mv("test_mv_src", "test_mv_dst")
    client.delete("test_mv_dst")

.. py:method:: list_dirs(fs_path)
列出本地路径下所有的子目录。

参数：
    - **fs_path** (str): 本地文件路径。

返回：
    - List: 该路径下所有的子目录名。

**示例代码**：

.. code-block:: python

    from paddle.distributed.fleet.utils import LocalFS

    client = LocalFS()
    subdirs = client.list_dirs("./")