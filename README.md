# OpenScan 复现操作包

论文：[OpenScan](https://arxiv.org/abs/2408.11030)，本地 PDF 为 `OpenScan-2408.11030v2.pdf`。

**Windows 用于编辑和上传 GitHub；Conda 环境、CUDA 编译和 GPU 推理在 Linux A40 服务器创建和执行。** 不要把 Windows 的虚拟环境复制到 Linux。

本包已整理操作步骤与辅助脚本，尚未在 A40 安装或运行；不是已验证可复现的镜像或依赖锁。建议先跑通 OpenMask3D，再扩展 SAI3D、MaskClustering、Open3DIS。官方 OpenScan 仓库包含这四个实例分割基线的适配；论文中的 OpenScene、PLA、RegionPLC 语义分割实验需要另外准备其官方代码，不能用本包四个基线的命令替代。

## 1. Windows 上传 GitHub

上传当前项目的 README、scripts、envs、.gitignore、.gitattributes 和论文 PDF 即可。`code/` 是嵌套的官方 Git 仓库，已经被忽略；服务器通过脚本重新获取固定提交，避免 Git 将嵌套仓库误当作只有引用、没有源码的目录。

在 GitHub 新建一个空仓库，再在当前文件夹 PowerShell 执行（替换 URL；如果已经初始化 Git，不要重复初始化）：

```powershell
git init
git add README.md .gitignore .gitattributes scripts envs OpenScan-2408.11030v2.pdf
git diff --cached --stat
git commit -m "Add OpenScan reproduction instructions and helpers"
git branch -M main
git remote add origin https://github.com/sh9adow/openscan_test.git
git push -u origin main
```

数据、模型权重、虚拟环境、日志与旧 evidence 已忽略，不上传。SSH 密码、密钥不写入文件。

## 2. VS Code 连到服务器并准备项目

在你已连接服务器的 VS Code 窗口打开终端。下面所有 bash 命令都在 **远端 Linux** 执行。先选一个有足够空间、你有写权限的目录：

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/sh9adow/openscan_test.git openscan-repro
cd openscan-repro
export PROJECT_ROOT="$PWD"
nvidia-smi
df -h .
which conda
```

服务器已有 Conda 就继续。没有则按 [Miniconda 官方说明](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)在自己的用户目录安装 Linux 版本；不用管理员权限，也不要覆盖其他人的安装。安装后新开终端。已有安装但提示 `conda: command not found`，按真实安装位置初始化，例如：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
# 如果实际安装在 ~/anaconda3，就改用该目录。
```

准备固定版本的代码及运行副本（会联网克隆源码，不运行模型）：

```bash
python scripts/prepare_source.py
export WORK="$PROJECT_ROOT/runs/openscan"
export OM="$WORK/Evaluation/OpenMask3D"
```

脚本固定官方提交 `28a14308a36f6b0d09951b1b2ed3bf4e5a66af95`。`code/OpenScan` 保持原样，后续配置和编译发生在 `runs/openscan`。再次运行会拒绝覆盖已有副本；需要另一份可用 `--work runs/openscan-02`。

## 3. 创建 OpenMask3D 虚拟环境

Conda 环境是 Python 和依赖的独立目录。名字只是标识，不是项目文件夹。创建一次，每次新终端使用前激活；退出用 `conda deactivate`。以下路线依据仓库旧版安装脚本整理：Python 3.8、PyTorch 1.12.1、CUDA runtime 11.3。不要直接用本机 Python 3.12 或最新版 PyTorch。

```bash
conda create -n openscan-om3d python=3.8 pip=23.3.2 -y
conda activate openscan-om3d
python -m pip install setuptools==68.0.0 wheel==0.41.3
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y
python -m pip install -r "$PROJECT_ROOT/envs/openmask3d-extra.txt"
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

最后应显示 CUDA 可用和服务器分配给你的 GPU。PyTorch 版本命令参考[官方历史版本页面](https://pytorch.org/get-started/previous-versions/)。额外依赖文件是安装起点，不是已求解的锁文件；若解析或构建失败，保留完整报错再修正版本。

**CUDA runtime 与编译器是两件事。** `nvidia-smi` 顶部的 CUDA Version 不是当前 nvcc 版本；Conda 的 `cudatoolkit` 也不保证提供 nvcc。接下来编译需要服务器已安装与 PyTorch CUDA 匹配的 Toolkit，以及可用的 GCC/G++、OpenBLAS：

```bash
nvcc --version
gcc --version
g++ --version
conda install -c anaconda openblas-devel -y
# 仅当管理员提供的真实路径是这里时使用：
export CUDA_HOME=/usr/local/cuda-11.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=4
export OMP_NUM_THREADS=3
# A40 的计算能力为 8.6；换 GPU 时重新核对。
export TORCH_CUDA_ARCH_LIST=8.6
```

若只有 CUDA 12.x Toolkit，先请管理员提供 CUDA 11.3 Toolkit/module，或另拟整套匹配环境；不要仅把 CUDA_HOME 指到不存在的目录。共享服务器的驱动不要自行替换。

逐个安装编译依赖，出错就停在当前步骤。不要直接执行上游整个安装脚本：它会切换目录，部分关键依赖还被注释掉。

```bash
python -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf'

mkdir -p "$PROJECT_ROOT/runs/build-deps"
cd "$PROJECT_ROOT/runs/build-deps"
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
git rev-parse HEAD > "$PROJECT_ROOT/runs/minkowskiengine-commit.txt"
python setup.py install --blas=openblas --blas_include_dirs="$CONDA_PREFIX/include" --force_cuda

cd "$OM/openmask3d/class_agnostic_mask_computation/third_party/pointnet2"
python -m pip install --no-build-isolation --no-deps .
python -m pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
python -m pip install --no-deps \
  'git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16'
python -m pip install --no-deps \
  'git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588'
export PYTHONPATH="$OM:$OM/openmask3d/class_agnostic_mask_computation:${PYTHONPATH:-}"
python -c "import torch, MinkowskiEngine, detectron2, pointnet2, clip, segment_anything, hydra; print('imports OK')"
python -m pip check
```

MinkowskiEngine 构建方式参考[官方说明](https://github.com/NVIDIA/MinkowskiEngine)。这里记录其实际 commit，尚未预先固定为验证通过版本。此 OpenScan 副本的 OpenMask3D 根目录没有 setup.py，因此不要照搬 README 的 `pip install -e .`；使用上述 PYTHONPATH。

在 VS Code 中选择远端解释器：命令面板 → `Python: Select Interpreter` → 选择 `openscan-om3d`。新终端重新设置 PROJECT_ROOT、WORK、OM 和 PYTHONPATH，或将这些配置保存到自己的启动脚本。

## 4. 数据和权重

需要同时具备：

- ScanNet200 验证场景：点云、RGB-D、相机位姿、内参、segments、官方预处理结果。按 [ScanNet 官方仓库](https://github.com/ScanNet/ScanNet)获得访问与下载方式。
- OpenScan 验证 GT 与标签映射：[作者链接的数据镜像](https://huggingface.co/datasets/garrying/OpenScan)。它提供标注，不能替代 ScanNet RGB-D 数据。
- ScanNet200 训练的 Mask3D checkpoint、SAM ViT-H checkpoint、CLIP ViT-L/14@336px。Mask3D 下载链接在运行副本的 OpenMask3D README 的 checkpoint 小节。不要混用单场景任意场景模型和论文 ScanNet200 模型。

建议目录：

```text
data/
  scans/                         # OpenMask3D README 要求的 ScanNet RGB-D 布局
  processed/scannet/             # 官方预处理结果，含 label_database.yaml
  openscan_val_gt/gt_material/   # 每个验证场景一个 txt
weights/
  scannet200_val.ckpt
  sam_vit_h_4b8939.pth
  ViT-L-14-336px.pt
outputs/
  openmask3d/masks/
  openmask3d/mask_features/
```

GT 可以在独立下载环境获取，避免给老模型环境强行升级 huggingface_hub：

```bash
conda create -n openscan-download python=3.11 -y
conda activate openscan-download
python -m pip install huggingface_hub
hf download garrying/OpenScan --repo-type dataset --local-dir "$PROJECT_ROOT/data" \
  --include 'openscan_val_raw.json' 'openscan_label_mapping/**' 'openscan_val_gt/**'
conda activate openscan-om3d
```

先阅读数据许可。所有下载都在你执行时产生网络与磁盘用量。SAM 权重来自官方发布地址：

```bash
mkdir -p "$PROJECT_ROOT/weights"
wget -O "$PROJECT_ROOT/weights/sam_vit_h_4b8939.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
python -c "import clip; clip.load('ViT-L/14@336px', device='cpu', download_root='$PROJECT_ROOT/weights')"
```

ScanNet 预处理遵循上游指令（路径需替换；不下载 ScanNet 就不能执行）：

```bash
cd "$OM/openmask3d/class_agnostic_mask_computation"
python -m datasets.preprocessing.scannet_preprocessing preprocess \
  --data_dir=/实际的原始ScanNet目录 \
  --save_dir="$PROJECT_ROOT/data/processed/scannet" \
  --git_repo=/实际的ScanNet官方代码目录 \
  --scannet200=true
```

## 5. OpenMask3D：掩码 → 特征 → OpenScan 评测

先使用一小部分场景验证数据和显存；小样本只能称为试跑，不能对照全量论文指标。官方单场景入口 `run_openmask3d_single_scene.sh` 默认 120 queries，且不使用完整验证集流程，不能冒充论文 150-query 全量协议。

在运行副本 `$OM/run_openmask3d_scannet200_eval.sh` 修改：

```text
SCANS_PATH                 = 绝对路径/data/scans
SCANNET_PROCESSED_DIR      = 绝对路径/data/processed/scannet
MASK_MODULE_CKPT_PATH      = 绝对路径/weights/scannet200_val.ckpt
SAM_CKPT_PATH              = 绝对路径/weights/sam_vit_h_4b8939.pth
OUTPUT_FOLDER_DIRECTORY   = 绝对路径/outputs/openmask3d
```

保留 `model.num_queries=150`；按你获得的 RGB-D 数据核对 `$OM/openmask3d/configs/openmask3d_scannet200_eval.yaml` 的内参分辨率、图像路径和深度尺度。该 bash 脚本后面有硬编码输出路径，要修改最终真正使用的赋值。**将脚本最后第 3 步 closed-set evaluation 注释掉，保留第 1、2 步**：原入口默认评 ScanNet200，不是 OpenScan。

```bash
conda activate openscan-om3d
cd "$OM"
mkdir -p "$PROJECT_ROOT/logs"
set -o pipefail
bash run_openmask3d_scannet200_eval.sh 2>&1 | tee "$PROJECT_ROOT/logs/openmask3d-generation.log"
```

完成后检查每个场景同时存在 `sceneXXXX_YY_masks.pt` 和 `sceneXXXX_YY_openmask3d_features.npy`。它们分别是点×掩码、掩码×768维特征；GT 必须保持相同点顺序。官方入口会跳过部分缺失特征，本包的材质评测器则要求全部 312 个验证场景齐备。

```bash
cd "$PROJECT_ROOT"
python -B scripts/evaluate_material.py \
  --source "$WORK" \
  --gt "$PROJECT_ROOT/data/openscan_val_gt/gt_material" \
  --masks "$PROJECT_ROOT/outputs/openmask3d/masks" \
  --features "$PROJECT_ROOT/outputs/openmask3d/mask_features" \
  --clip-checkpoint "$PROJECT_ROOT/weights/ViT-L-14-336px.pt" \
  --output "$PROJECT_ROOT/outputs/material-eval-001"
```

输出为 inputs.json（输入与评测源码哈希）、evaluation.log、metrics.json。材质查询固定为 `this term is made of MATERIAL`；根据原始评测入口的输出逻辑保持每个掩码置信分数为 1.0，不自行替换为另一种评分。

论文 v2 Table 4 / Table 8 的 **OpenMask3D 材质属性**对照为 AP=18.8、AP50=22.1、AP25=25.0，单位百分数。它们不是全部八方面的 Mean。需确认掩码生成、权重、分割划分和随机设置一致，再比较运行值；脚本运行成功本身不代表论文复现成功。

其他七方面需要分别选择 `openscan_constants_<aspect>.py`、`eval_semantic_instance_<aspect>.py`、对应 GT 与论文查询模板，不能只改 gt 路径。当前辅助评测脚本仅实现 material，其余方面尚未封装。

## 6. 另外三个实例分割基线

分别建环境，不共用 OpenMask3D 的 Python/PyTorch。下面是阶段顺序；权重、数据路径与编译步骤以各目录 README 为依据，尚未在本机执行。

### SAI3D

```bash
conda create -n openscan-sai3d python=3.8 pip=23.3.2 -y
conda activate openscan-sai3d
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install open3d natsort matplotlib tqdm opencv-python scipy plyfile
cd "$WORK/Evaluation/SAI3D"
```

按该目录 README 安装 Semantic-SAM、其 Detectron2 与 CUDA ops，并下载 `swinl_only_sam_many2many.pth`。README 混用了 `Semantic-SAM` 和 `Semantic_SAM`，以实际 clone 目录为准。配置 scripts 中的路径和 GPU 后依次：

```bash
bash scripts/sam_scannet.sh
bash scripts/seg_scannet.sh
python helpers/format_convertion.py --app=1 \
  --base_dir=/实际的SAI3D掩码输出目录 --out_dir=/实际的pt掩码目录
```

随后激活 `openscan-om3d`，使用 `$OM/run_openmask3d_scannet200_eval_sai3d.sh` 的第 1 步为 SAI3D 掩码提取 CLIP 特征，手动修改该脚本路径，跳过最后的默认 ScanNet200 评测。再用上面的 material 评测入口，将 masks/features 指向 SAI3D 的输出即可。SAI3D 目录也有 `scripts/run_openmask3d_scannet200.sh`，其 BASE_DIR 及 `cd openmask3d/openmask3d` 是作者机器布局，不能原样运行。

### MaskClustering

```bash
conda create -n openscan-maskclustering python=3.8 pip=23.3.2 -y
conda activate openscan-maskclustering
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
cd "$WORK/Evaluation/MaskClustering"
python -m pip install -r requirements.txt
```

还需按该目录 README 编译 PyTorch3D、Detectron2/CropFormer 的 CUDA ops，安装 open_clip_torch 并准备 CropFormer、CLIP ViT-H/14 权重。未固定的新版依赖可能不再支持 Python 3.8，安装失败时按仓库时期选兼容版本并记录，不要盲目升级整个环境。

将 `configs/scannet.json` 的 cropformer_path、数据读取/GT 路径和 `run.py` 中的 CUDA_LIST 改为服务器实际配置；按 README 准备 ScanNet RGB-D。先跑 demo 检查流水线，再执行：

```bash
python run.py --config scannet
```

正式 OpenScan 评测需在运行副本选择 `evaluation/constants_<aspect>*.py`，并使用对应 OpenScan GT。检查查询是否使用论文模板；`scannet` 配置名本身不意味着评测了 OpenScan。

### Open3DIS

```bash
conda create -n openscan-open3dis python=3.8 pip=23.3.2 -y
conda activate openscan-open3dis
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -y
python -m pip install spconv-cu113==2.1.25
python -m pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
cd "$WORK/Evaluation/Open3DIS"
```

继续按 `docs/INSTALL.md` 编译 Detectron2、本地 GroundingDINO、segment_anything、ISBNet 和 pointnet2，再安装 requirements。ISBNet 编译需要系统 libsparsehash-dev，无 sudo 权限时找管理员提供。按 `docs/DATA.md` 准备点云、superpoints、图像和预训练权重。

修改 `configs/scannet200.yaml` 中的所有实际路径。**检查每个 scripts/*.sh 的 CUDA_VISIBLE_DEVICES，仓库有硬编码为 1 的脚本；单张分配 GPU 需改成可见设备中的正确编号。** 按 `docs/RUN.md`：

```bash
sh scripts/grounding_2d.sh configs/scannet200.yaml
sh scripts/generate_3d_inst.sh configs/scannet200.yaml
sh scripts/refine_grounding_feat.sh configs/scannet200.yaml
sh scripts/generate_3d_inst.sh configs/scannet200.yaml
# 选择 OpenScan 属性标签，改好 eval.py 的 data_path 与 gt_txt_base_path 后：
sh scripts/eval.sh
```

OpenScan 属性标签变体位于 `open3dis/dataset/`。论文设置是 Grounded-SAM + ISBNet；默认配置若使用其他 2D 模型或只用 2D proposals，需要先对齐，不能直接对照论文行。

## 7. 留存记录与常见问题

安装成功后导出 Linux 环境，保存命令、日志和修改：

```bash
mkdir -p "$PROJECT_ROOT/outputs/environment"
conda env export --no-builds > "$PROJECT_ROOT/outputs/environment/conda.yml"
python -m pip freeze > "$PROJECT_ROOT/outputs/environment/pip-freeze.txt"
nvidia-smi > "$PROJECT_ROOT/outputs/environment/nvidia-smi.txt"
nvcc --version > "$PROJECT_ROOT/outputs/environment/nvcc.txt"
```

- `invalid metadata` / OmegaConf 无法安装：使用文档的 pip 23.3.2；旧 Lightning/OmegaConf 与新 pip 不兼容。
- `CUDA_HOME`、nvcc not found、CUDA mismatch：核对 Toolkit 与 torch.version.cuda；只安装 CUDA runtime 不够编译扩展。
- `undefined symbol`：检查扩展是否用当前 Python、PyTorch、CUDA 编译；切换环境后通常要重编译。
- `pip install -e .` 找不到 setup.py：此 OpenMask3D 副本使用 PYTHONPATH，不在根目录做 editable install。
- `No module named openmask3d/models/datasets`：检查工作目录和本文 PYTHONPATH；不要随手 pip 安装同名第三方包。
- 缺少 GT、masks、features：先完成数据准备与生成，不能用零值或模拟结果填充。
- 所有 Linux 命令、GPU 扩展与模型推理均尚未在 A40 验证。本包只完成本地语法和入口检查。

复现结论仅适用于记录的环境、协议与计算主张；材质属性的单基线结果不代表整篇论文复现。
