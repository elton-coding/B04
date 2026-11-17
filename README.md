# B04 GPU 开发

本仓库当前包含 CUDA 自检、A/B 类观测输入发现、字段归一化、数据检查，以及 GPU 脉冲特征预处理。运行 `--task audit` 会生成 `input_audit.json`。运行 `--task features` 会分批生成保留脉冲标识的 Parquet，其中包含原观测和 GPU 计算的方位正弦、余弦。五项预测算法和正式提交结果仍在开发。

## 本机位置

| 内容 | 路径 |
|---|---|
| Windows 项目与代码 | `D:\Source\B04`，Python 源码在 `src\` |
| WSL 项目目录 | `/mnt/d/Source/B04` |
| WSL 项目虚拟环境 | `~/.venvs/b04` |
| WSL Docker 引擎配置 | `/etc/docker/daemon.json` |
| NVIDIA 容器运行时配置 | `/etc/nvidia-container-runtime/config.toml` |
| NVIDIA 软件源 | `/etc/apt/sources.list.d/nvidia-container-toolkit.list` |
| 项目镜像定义与评测入口 | `Dockerfile`、`requirements.txt`、`run.sh` |

在 PowerShell 进入指定的 Ubuntu 发行版。

```powershell
wsl -d Ubuntu-24.04 --cd /mnt/d/Source/B04
```

在 Ubuntu 中建立或查看项目环境。

```bash
uv venv --python /usr/bin/python3.12 ~/.venvs/b04
uv pip sync --python ~/.venvs/b04/bin/python requirements.txt
source ~/.venvs/b04/bin/activate
which python
python --version
python -m site
/usr/lib/wsl/lib/nvidia-smi
nvidia-ctk --version
docker run --rm --gpus all python:3.12-slim nvidia-smi
bash run.sh --task gpu-check
```

自检会执行一次实际的 CUDA 前向、反向传播及参数更新，并报告 CUDA 构建版本、显卡和计算能力。Ubuntu 的 NVIDIA Container Toolkit 使用 [NVIDIA 官方安装步骤](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)配置已有 Docker Engine。按赛题模板使用 CUDA 12.8 的 PyTorch 构建。

更新依赖时先修改 `requirements.in`，再在 Ubuntu 中执行 `uv pip compile requirements.in --emit-index-url -o requirements.txt` 生成锁定的版本。开发环境与镜像共用这份依赖清单。

## 验证数据与镜像

设置 `data_dir` 指向只包含选定比赛阶段数据的目录。程序通过 CSV 字段和首条记录的数据版本查找 A/B 表。测试集传入 `--test`，训练集省略该参数。

```bash
data_dir='/mnt/d/Source/B04/B04赛题初赛A榜测试数据-11月12日上传/B04赛题初赛A榜测试数据'
bash run.sh --task audit --test --data-root "$data_dir" --out-dir output/audit --limit 10000
bash run.sh --task features --test --data-root "$data_dir" --out-dir output/features --limit 10000

docker build -t b04-dev .
docker run --rm --gpus all b04-dev --task gpu-check
mkdir -p output/container-audit
docker run --rm --gpus all \
  --mount "type=bind,source=$data_dir,target=/input,readonly" \
  --mount "type=bind,source=$PWD/output/container-audit,target=/output" \
  --user "$(id -u):$(id -g)" b04-dev --task audit --test --limit 10000
```

去掉 `--limit` 可扫描完整输入。输入数据、主办方附件、运行产物、模型权重和凭据位于 Git 跟踪范围之外。结果目录默认为 `/output`，容器按官方约定从只读 `/input` 读取数据。

本机 WSL 直连部分软件源会超时。若出现该情况，可使用本机已有的代理 `http://127.0.0.1:10808`。代理配置属于本机环境，项目镜像可按比赛环境的联网方式构建。

```bash
proxy=http://127.0.0.1:10808
HTTPS_PROXY="$proxy" HTTP_PROXY="$proxy" uv pip sync --python ~/.venvs/b04/bin/python requirements.txt
docker build --network=host --build-arg HTTP_PROXY="$proxy" --build-arg HTTPS_PROXY="$proxy" -t b04-dev .
```

## Git

远端为 `https://github.com/elton-coding/B04.git`。首次初始化使用 `main`，后续可按可独立验证的功能建分支，并在合并前复跑自检和真实附件样本。

```bash
git status --short --branch
git remote -v
git log --oneline -8
```
