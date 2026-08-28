# SakuraMedia SigLIP2 Embedding Service

为 SakuraMedia 图片搜索提供 SigLIP2 图像与文本 embedding。镜像内已包含预先导出的模型和运行时：启动时不下载模型、不挂载模型目录，也不包含 PyTorch。

服务固定使用 `siglip2-base-patch16-224-v3` 向量空间，返回 L2 归一化的 768 维向量。请勿试图替换模型文件；更换模型或向量空间后，SakuraMedia 中已有的图片搜索索引不能继续混用。

图片接口接受静态 WebP 与 JPEG（`.jpg`/`.jpeg`），按文件内容识别，不依赖上传文件名。服务默认允许最高 4000 万像素的源图片，覆盖常见的 8K 横向图片；这是防止异常图片耗尽内存的准入上限，不是模型输入尺寸。libwebp 会在解码阶段直接输出模型需要的 224×224 RGB。JPEG 则先由 libjpeg-turbo 在解码阶段按 DCT 缩小到最多 100 万 RGB 像素，再缩放为 224×224，因此不会在内存中展开完整大图。

渐进式 JPEG 需要解码器保留整张图的系数，不能像基线 JPEG 一样完全按行流式处理；为保持内存边界，服务支持最多 1000 万源像素、32 个扫描的渐进式 JPEG。这个额外上限只影响渐进式 JPEG，不影响基线 JPEG、WebP 或 `MAX_IMAGE_PIXELS` 的 4000 万像素准入上限。图片 CPU 预处理并发数由 `CPU_CONCURRENCY` 控制，默认值为 1。可通过 `MAX_IMAGE_PIXELS` 调整通用源图片准入上限。

## 镜像选择

| 镜像 | 平台 | 用途 |
| --- | --- | --- |
| `tinyping/siglip2-embed-service:cpu` | Linux amd64、arm64 | 纯 CPU。适用于 AMD CPU、Intel CPU 和 ARM64 设备。 |
| `tinyping/siglip2-embed-service:intel` | Linux amd64 | Intel 核显。使用 OpenVINO，需将 `/dev/dri` 传入容器。 |
| `tinyping/siglip2-embed-service:cuda` | Linux amd64 | NVIDIA 显卡。使用 CUDA，需要 NVIDIA Container Toolkit。 |
