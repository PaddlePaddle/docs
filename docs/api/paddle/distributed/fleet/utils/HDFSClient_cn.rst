.. _cn_api_distributed_fleet_utils_fs_HDFSClient:

HDFSClient
-------------------------------

.. py:class:: paddle.distributed.fleet.utils.HDFSClient
一个 HADOOP 文件系统工具类。

参数
::::::::::::

    - **hadoop_home** (str)：HADOOP HOME 地址。
    - **configs** (dict): HADOOP 文件系统配置。需包含 `fs.default.name` 和 `hadoop.job.ugi` 这两个字段。

代码示例
::::::::::::

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient

方法
::::::::::::
ls_dir(fs_path)
'''''''''
列出 `fs_path` 路径下所有的文件和子目录。

**参数**

    - **fs_path** (str): HADOOP 文件路径。

**返回**

    - Tuple，一个包含所有子目录和文件名的 2-Tuple，格式形如：([subdirname1, subdirname1, ...], [filename1, filename2, ...])。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.ls_dir

mkdirs(fs_path)
'''''''''
创建一个目录。

**参数**

    - **fs_path** (str): HADOOP 文件路径。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.mkdirs

delete(fs_path)
'''''''''
删除 HADOOP 文件（或目录）。

**参数**

    - **fs_path** (str): HADOOP 文件路径。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.delete

is_file(fs_path)
'''''''''
判断当前路径是否是一个文件。

**参数**

    - **fs_path** (str): HADOOP 文件路径。

**返回**

    - Bool：若当前路径存在且是一个文件，返回 `True`，反之则返回 `False` 。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.is_file

is_dir(fs_path)
'''''''''
判断当前路径是否是一个目录。

**参数**

    - **fs_path** (str): HADOOP 文件路径。

**返回**

    - Bool：若当前路径存在且是一个目录，返回 `True`，反之则返回 `False` 。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.is_dir

is_exist(fs_path)
'''''''''
判断当前路径是否存在。

**参数**

    - **fs_path** (str): HADOOP 文件路径。

**返回**

    - Bool：若当前路径存在返回 `True`，反之则返回 `False` 。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.is_exist

upload(local_path, fs_path)
'''''''''
上传本地文件至 HADOOP 文件系统。

**参数**

    - **local_path** (str)：本地文件路径。
    - **fs_path** (str): HADOOP 文件路径。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.upload

download(fs_path, local_path)
'''''''''
下载 HADOOP 文件至本地文件系统。

**参数**

    - **local_path** (str)：本地文件路径。
    - **fs_path** (str): HADOOP 文件路径。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.download

touch(fs_path, exist_ok=True)
'''''''''
创建一个 HADOOP 文件。

**参数**

    - **fs_path** (str): HADOOP 文件路径。
    - **exist_ok** (bool)：路径已存在时程序是否报错。若 `exist_ok = True`，则直接返回，反之则抛出文件存在的异常，默认不抛出异常。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.touch

mv(fs_src_path, fs_dst_path, overwrite=False)
'''''''''
HADOOP 系统文件移动。

**参数**

    - **fs_src_path** (str)：移动前源文件路径名。
    - **fs_dst_path** (str)：移动后目标文件路径名。
    - **overwrite** (bool)：若目标文件已存在，是否删除进行重写，默认不重写并抛出异常。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.mv

list_dirs(fs_path)
'''''''''
列出 HADOOP 文件路径下所有的子目录。

**参数**

    - **fs_path** (str): HADOOP 文件路径。

**返回**

    - List：该路径下所有的子目录名。

**代码示例**

COPY-FROM: paddle.distributed.fleet.utils.HDFSClient.list_dirs
