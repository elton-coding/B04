# B04 GPU 开发

本仓库支持 CUDA 自检、A/B 类输入审计、GPU 脉冲特征预处理、本地留出评估，以及初版竞赛候选推理。运行 `--task audit` 会生成 `input_audit.json`。运行 `--task features` 会分批生成保留脉冲标识的 Parquet，其中包含原观测和 GPU 计算的方位正弦、余弦。

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

赛题说明列出五项机器评分维度。公开附件未给出评分器源码、各项公式或权重。A 榜按最新一次成绩排名，截止时前 30 名进入 B 榜评测。初赛晋级综合分为 B 榜机器分的 80% 加专家分的 20%。下列本地指标用于比较算法版本，官方 A 榜成绩以在线评测为准。

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
| 跨平台同源融合 | 全局对象 B³、跨平台脉冲对的精确率/召回率/F1，以及各平台组合的同口径分解 |
| 对象–航迹关联 | 预测对象与真值对象的脉冲集合 IoU 严格大于 0.5 时对齐。检查所连航迹的目标编号，报告匹配精确率、召回率、F1 和 `no_match` 召回率 |

宏 F1 对留出集实际出现的标签等权平均。没有分母的指标写为 JSON `null`。平台对在真值和预测中都没有同源脉冲对时省略。评估器用 SQLite 临时文件处理脉冲连接，系统临时目录需要可写空间。固定留出用于同场景的时间外推比较，公开 A 榜测试集和跨场景表现需要单独验证。

可另建 `output/robustness-mid/holdout`，复测 2026-11-10 09:04:50 至 09:06:30 这一时间窗。把切分命令的起止时间和输出目录换掉，再用该切分的 `train/A.csv` **重新训练**。默认窗口的模型已用复测窗的数据训练，直接复用权重会造成评估泄漏。

## CUDA 时序与航迹关联候选

候选在 32 ms 和 256 ms 两个时间窗内，按平台、粗频带和方位分组，提取脉冲参数的均值、标准差、密度与相邻到达间隔。CUDA 分类网络预测每条脉冲的型号和状态。训练时提高跟踪状态的采样比例并对错误加权，推理时用训练状态频数部分抵消采样和加权造成的状态概率偏移。训练标签用于监督，留出标签用于本地评分与选择判决参数。

推理时将 B 航迹位置按时刻插值，用 A 平台位置计算到各航迹的方位角。在方位测量相对明确的脉冲上估计航迹型号分布，再结合圆周方位误差、时间覆盖和型号相容性选择航迹。满足条件的脉冲按航迹编号归为对象，未关联脉冲暂按型号归组。后者需要继续解决同型号无航迹目标的分离。

使用上节生成的固定切分，在 WSL 训练和评估。

```bash
source ~/.venvs/b04/bin/activate
holdout=output/eval-reference/holdout
run=output/temporal-reference
python src/temporal_model.py --train-a "$holdout/train/A.csv" --output "$run/temporal.pt"
bash run.sh --task all --data-root "$holdout/input" --model "$run/temporal.pt" \
  --out-dir "$run/predictions"
python src/evaluate.py --truth-a "$holdout/truth/A.csv" --truth-b "$holdout/truth/B.csv" \
  --pulse-predictions "$run/predictions/P1A_test_pulse_predictions.csv" \
  --object-track-links "$run/predictions/P1A_test_object_track_links.csv" \
  --output "$run/report.json"
```

切换到官方 A 榜输入前，可在验证通过后用全部带标签训练脉冲重新训练权重。对无标签数据，把评估命令的 `--truth-a`、`--truth-b` 换成 `--input-a`、`--input-b`，分别传入官方 A、B 原始测试 CSV。该模式检查全部脉冲和关联记录的结构，并报告脉冲、对象和匹配航迹数量。建议每次打榜前保留 Git 提交号、权重、留出指标和待提交结果包。上述命令只写本地文件。

构建候选镜像时，将选中的网络权重放到项目根目录下的 `model/temporal.pt`，再按上节的 Docker 命令构建。项目镜像构建只纳入这一个权重文件，Git 忽略本地权重。镜像默认运行 `--task all` 并从 `/input` 读取测试 CSV，输出命名为 `<scenario_id>_<data_version>_*.csv` 的两张结果表到 `/output`。也可以在开发时将权重只读挂载到容器，显式传给 `--model`。

## Git

远端为 `https://github.com/elton-coding/B04.git`。首次初始化使用 `main`，后续可按可独立验证的功能建分支，并在合并前复跑自检和真实附件样本。

```bash
git status --short --branch
git remote -v
git log --oneline -8
```
