# A D 文字感知模块

公开仓库：https://github.com/wentengsong627-sudo/SAM_AD_Perception

下载后直接在仓库根目录运行下面的命令。

输入一句目标描述和连续的图像，输出该目标的可见掩码、目标框和跟踪状态。供同学在**自己的 NVIDIA 显卡电脑**上部署，也提供替换 FlyCo 感知实现的适配器。

| 方案 | 模型 | 连续跟踪方式 | 默认地址 |
| --- | --- | --- | --- |
| A | Grounding DINO Tiny + SAM2.1 Hiera Tiny | 文字定位后用 SAM2 视频记忆跟踪，异常时搜索和复核 | `http://127.0.0.1:8850` |
| D | EfficientSAM3 TinyViT 11M + MobileCLIP S0 | 每帧文字分割，再做应用层身份验证 | `http://127.0.0.1:8851` |

两套方案使用相同接口。当前交付接口每个进程只处理**一个活动会话和一个选定目标**。没有可靠可见目标时返回空掩码，保留状态供上层判断。

## 先读哪份文档

- **第一次接手：**[通俗接入指南](docs/接入指南.md)，也可下载 [Word 说明书](docs/A和D感知模块通俗接入说明.docx)。
- **按字段开发：**[OpenAPI 接口定义](openapi.json)。服务启动后也可打开 `/docs`。
- **要接原论文工程：**先阅读指南的 FlyCo 部分，再运行 `flyco_adapter/install_into_flyco.py`。

## 在自己的电脑启动

需要 NVIDIA 驱动和 Python 3.12。第一次安装和准备模型需要联网；之后推理可离线。大权重和第三方模型源码由 `prepare_assets.py` 从锁定来源获取并逐文件校验，避免把旧实验、缓存和大二进制提交到 Git。

**Windows PowerShell，在本文件所在目录：**

```powershell
.\setup_windows.ps1
.\start_A.ps1
```

**Ubuntu，另有 Python 3.12：**

```bash
bash setup_linux.sh
bash start_A.sh
```

另开终端，使用自己的视频：

```powershell
.\.venv\Scripts\python.exe examples/video_client.py --backend A --video "D:\videos\dog.mp4" --text dog --display
```

Ubuntu 对应命令的解释器改为 `.venv/bin/python`。换 D 时运行 `start_D`，示例改为 `--backend D`，其余调用相同。测试 A 后先关闭视频示例并在 A 服务终端按 Ctrl+C，再启动 D，避免同时占用显存。

## 接入自己的程序

```python
from sam_ad import Client

with Client("http://127.0.0.1:8850", backend="A") as sam:
    sam.wait_ready()
    sam.start("dog")                      # 一次任务只初始化一次
    for frame_bgr in your_video_frames:
        result = sam.process_bgr(frame_bgr)
        if result.visible:
            mask = result.mask            # H×W，uint8，0/255
            box = result.bbox_xyxy        # 右边界和下边界不包含在框内
            use_visible_observation(mask, box)
        else:
            handle_no_observation(result.states)
```

上面 `your_video_frames`、`use_visible_observation` 和 `handle_no_observation` 是你自己系统的数据来源和处理函数。可运行的完整例子在 [examples/video_client.py](examples/video_client.py)。调用方可单独安装 `sdk/` 下的客户端 wheel，不需要在 ROS 环境安装模型依赖。

## 交付范围

仅保留 A/D 的 CUDA 感知代码、统一 SDK、接口定义、必要的依赖准备脚本、FlyCo 适配器和回归检查。B/C、旧网页、Unreal 演示、RK3588 NPU 实验、历史结果和虚拟环境不在交付范围。第三方许可证见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。

Windows NVIDIA 的实际回放检查和精确边界见 [validation.json](validation.json)。接口测试通过不等于分割准确率通过；完整 ROS、三维预测、Unreal 与飞行闭环仍需在接收方原系统验收。
