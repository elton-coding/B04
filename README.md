# B04 GPU 开发

本仓库支持 CUDA 自检、A/B 类输入审计、GPU 脉冲特征预处理，以及本地留出评估。运行 `--task audit` 会生成 `input_audit.json`。运行 `--task features` 会分批生成保留脉冲标识的 Parquet，其中包含原观测和 GPU 计算的方位正弦、余弦。正式提交入口仍在开发。

## 本机位置

| 内容 | 路径 |
|---|---|
| Windows 项目与代码 | `D:\Source\B04`，Python 源码在 `src\` |
| WSL 项目目录 | `/mnt/d/Source/B04` |
| WSL 项目虚拟环境 | `~/.venvs/b04` |
| 本机依赖轮子缓存 | `~/.cache/b04-wheels` |
| WSL Docker 引擎配置 | `/etc/docker/daemon.json` |
| NVIDIA 容器运行时配置 | `/etc/nvidia-container-runtime/config.toml` |
| NVIDIA 软件源 | `/etc/apt/sources.list.d/nvidia-container-toolkit.list` |
| 项目镜像定义与评测入口 | `Dockerfile`、`requirements.txt`、`run.sh` |

在 PowerShell 进入指定的 Ubuntu 发行版。

```powershell
wsl -d Ubuntu-24.04 --cd /mnt/d/Source/B04
```

首次创建环境，在 Ubuntu 执行以下命令。

```bash
uv venv --python /usr/bin/python3.12 ~/.venvs/b04
uv pip sync --python ~/.venvs/b04/bin/python requirements.txt
```

查看现有环境和运行自检。

```bash
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

本机已缓存完整依赖，可在 Ubuntu 中用下列命令离线复建开发镜像。同一份代码和依赖清单仍使用根目录的 Dockerfile 作为提交构建定义。

```bash
mkdir -p .cache
sed 's|^RUN pip install --no-cache-dir -r requirements.txt$|RUN --mount=type=bind,from=b04_wheelhouse,target=/wheels pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt|' \
  Dockerfile > .cache/Dockerfile.b04-offline
docker build --network=none --build-context b04_wheelhouse="$HOME/.cache/b04-wheels" \
  -f .cache/Dockerfile.b04-offline -t b04-dev .
```

## 本地留出评估

赛题说明列出五项机器评分维度。公开附件未给出评分器源码、各项公式或权重。这里复算的是本地验证指标，官方 A 榜成绩以在线评测为准。

使用初赛 A 榜带标签的 A/B 训练数据，按同一观测时间切分。固定验证窗口为 2026-11-10 09:06:40 至 09:08:23，含起点、排除终点。训练只用 09:06:30 之前的数据，中间保留 10 秒隔离。生成的 `input` 剥离目标标签供推理，`truth` 只供评分。比较版本时使用同一个 `split.json` 和同一份留出数据。

在 WSL 项目目录执行以下命令。生成切分和基线的目标目录需为空缺路径。

```bash
source ~/.venvs/b04/bin/activate
data_dir='/mnt/d/Source/B04/B04赛题初赛A榜训练数据-11月12日上传/B04赛题初赛A榜训练数据'
run=output/eval-reference
python src/make_holdout.py --data-root "$data_dir" --out-dir "$run/holdout" \
  --validation-start 2026-11-10T09:06:40 --validation-end 2026-11-10T09:08:23 --gap-seconds 10
python src/local_baseline.py --holdout "$run/holdout" --out-dir "$run/baselines"

for model in majority gaussian; do
  python src/evaluate.py --truth-a "$run/holdout/truth/A.csv" --truth-b "$run/holdout/truth/B.csv" \
    --pulse-predictions "$run/baselines/$model/pulse_predictions.csv" \
    --object-track-links "$run/baselines/$model/object_track_links.csv" \
    --output "$run/$model.json"
done
```

多数类基线用训练集中最常见的型号和状态作预测。CUDA 对角高斯基线用 `param1`、`param2`、`param3` 分别预测每个脉冲的型号和状态。两条基线都按预测型号归组，并把所有对象标记为 `no_match`，供后续实例级聚类和航迹匹配算法做对照。

把评估命令里的两个预测路径替换成新算法的结果，即可复用同一真值。程序先检查列顺序、脉冲覆盖、枚举值、对象覆盖和航迹唯一性，再输出各项指标。

| 赛题维度 | 当前本地指标 |
|---|---|
| 平台内脉冲分选 | 按平台计算成员重合的 B³ 精确率、召回率、F1 |
| 目标型号 | 脉冲准确率、宏 F1、逐型号统计、混淆矩阵 |
| 工作状态 | 脉冲准确率、宏 F1、逐状态统计、混淆矩阵 |
| 跨平台同源融合 | 全局对象 B³，以及只统计跨平台脉冲对的精确率、召回率、F1 |
| 对象–航迹关联 | 预测对象与真值对象的脉冲集合 IoU 严格大于 0.5 时对齐。检查所连航迹的目标编号，报告匹配精确率、召回率、F1 和 `no_match` 召回率 |

宏 F1 对留出集实际出现的标签等权平均。没有分母的指标写为 JSON `null`。评估器用 SQLite 临时文件处理脉冲连接，系统临时目录需要可写空间。固定留出用于同场景的时间外推比较，公开 A 榜测试集和跨场景表现需要单独验证。

## Git

远端为 `https://github.com/elton-coding/B04.git`。首次初始化使用 `main`，后续可按可独立验证的功能建分支，并在合并前复跑自检和真实附件样本。

```bash
git status --short --branch
git remote -v
git log --oneline -8
```
