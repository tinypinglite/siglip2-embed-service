# SakuraMedia SigLIP2 Embedding Service

为 SakuraMedia 图片搜索提供 SigLIP2 图像与文本 embedding。镜像内已包含预先导出的模型和运行时：启动时不下载模型、不挂载模型目录，也不包含 PyTorch。

服务固定使用 `siglip2-base-patch16-224-v1` 向量空间，返回 L2 归一化的 768 维向量。请勿试图替换模型文件；更换模型或向量空间后，SakuraMedia 中已有的图片搜索索引不能继续混用。

服务默认允许最高 4000 万像素的源图片，覆盖常见的 8K 横向图片；这是防止异常图片耗尽内存的准入上限，不是模型输入尺寸。服务会串行完成源图片解码和模型预处理，再由模型自带的 processor 缩放到 224×224。调用方不需要预先缩放。可通过 `MAX_IMAGE_PIXELS` 调整源图片准入上限。

## 镜像选择

| 镜像 | 平台 | 用途 |
| --- | --- | --- |
| `tinyping/siglip2-embed-service:cpu` | Linux amd64、arm64 | 纯 CPU。适用于 AMD CPU、Intel CPU 和 ARM64 设备。 |
| `tinyping/siglip2-embed-service:intel` | Linux amd64 | Intel 核显。使用 OpenVINO，需将 `/dev/dri` 传入容器。 |
| `tinyping/siglip2-embed-service:cuda` | Linux amd64 | NVIDIA 显卡。使用 CUDA，需要 NVIDIA Container Toolkit。 |
