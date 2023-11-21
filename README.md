# EDTER
> [EDTER: Edge Detection with Transformer](https://arxiv.org/abs/2203.08566)                 
> Mengyang Pu, Yaping Huang, Yuming Liu, Qingji Guan and Haibin Ling                 
> *CVPR 2022*

**🔥Update**<br/>
*More detailed usage*<br/>
*The comparison of the reported results and the reproduced results*<br/>
*All training logs, including the experimental environment*

## Contents
```
0 Issues and Answers
1 Usage
    1.1 Linux
    1.2 Datasets
    1.3 Initial weights
2 Traning
    2.1 Step1: The training of EDTER-Stage I on BSDS500
    2.2 Step2: The training of EDTER-Stage II on BSDS500
    2.3 How to train the EDTER model on BSDS-VOC (BSDS500 and PASCAL VOC Context):
        Step1: The training of EDTER-VOC-Stage I on PASCAL VOC Context
    2.4 Step2: The training of EDTER-VOC-Stage I on BSDS500
    2.5 Step3: The training of EDTER-VOC-Stage II on BSDS500
3 Testing
    3.1 EDTER-Stage I with single-scale testing
    3.2 EDTER-Stage I with multi-scale testing
    3.4 EDTER-Stage II with single-scale testing
    3.5 EDTER-Stage II with multi-scale testing
4 🔥🔥The comparison of the reported results and the reproduced results🔥🔥
    4.1 The results of EDTER-Stage I on BSDS500
    4.2 The results of EDTER-Stage II on BSDS500
    4.3 The EDTER model pre-trained on the PASCAL VOC Context dataset
    4.4 The results of EDTER-VOC-Stage I on BSDS500
    4.5 The results of EDTER-VOC-Stage II on BSDS500
5 Eval
6 Results
7 Download the Pre-trained model
Important notes
Acknowledgments
Reference
```
## Issues and Answers
**🔥Q:** How to change batch_size?<br/>
**🔥A:** **the batch size of training = samples_per_gpu * GPU_NUM**
If you want to set *samples_per_gpu*, please refer to
https://github.com/MengyangPu/EDTER/blob/bbee219d5713a77aeec61c0f7fde93620cb02d60/configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py#L99
For example, *data = dict(samples_per_gpu=4)* means that each GPU can process 4 images. If training batch_size=8, please set samples_per_gpu=4 and GPU_NUM=2.

**🔥Q:** KeyError: 'BSDSDataset is not in the dataset registry'.<br/>
**🔥A:** <br/>
```
cd EDTER
pip install -e .  # or "python setup.py develop"
pip install -r requirements/optional.txt
```

**🔥Q:** Dataset download.<br/>
**🔥A:** Please refer to [1.2 Datasets](https://github.com/MengyangPu/EDTER/#12-datasets)

**🔥🔥Q:** Reproduced results.<br/>
**🔥🔥A:** Please refer to [4 The comparison of the reported results and the reproduced results](https://github.com/MengyangPu/EDTER#4-the-comparison-of-the-reported-results-and-the-reproduced-results), and we upload all reproduced results on BaiDuNetdisk.<br/>
❗Note: The capacity of our Google Drive is limited, and all training files (including .log files, .mat files, .png files, and .pth files) for each model are approximately 20GB, so we upload them to BaiDuNetdisk. If you cannot download it, please contact me (email:mengyang.pu@ncepu.edu.cn).<br/>
|     Reproduced results   |                 Download             |
| -------------------------| ------------------------------------ |
| [EDTER-Stage I](https://github.com/MengyangPu/EDTER/#41-the-results-of-edter-stage-i-on-bsds500) | [BaiDuNetdisk](https://pan.baidu.com/s/158B9xct-J8nnOBGSPuotRA?pwd=nx35) or [Google Drive](https://drive.google.com/drive/folders/1vwX_gAmhCvJwbMEGGO1Hh63-xrWMzgXA?usp=sharing)|
| [EDTER-Stage II](https://github.com/MengyangPu/EDTER/#42-the-results-of-edter-stage-ii-on-bsds500) | [BaiDuNetdisk](https://pan.baidu.com/s/1JzlXAH8YnOEFiDncjSDZpA?pwd=mawm) or [Google Drive](https://drive.google.com/drive/folders/1I9C_FeV1hPM3lZzdAOkKSyeSl9OBw3Mm?usp=sharing)|
| [EDTER-VOC-Stage I pre-train](https://github.com/MengyangPu/EDTER/#43-the-edter-model-pre-trained-on-the-pascal-voc-context-dataset) | [BaiDuNetdisk](https://pan.baidu.com/s/1SS62jBW-Qao7BQ3nXrDvYQ?pwd=dk5v) or [Google Drive](https://drive.google.com/drive/folders/1pFCRjHfD-Jpnxn0STOouhGM4zpof2dGe?usp=sharing)|
| [EDTER-VOC-Stage I](https://github.com/MengyangPu/EDTER/#44-the-results-of-edter-voc-stage-i-on-bsds500) | [BaiDuNetdisk](https://pan.baidu.com/s/15CIuL2r0fZckSifgNFanBw?pwd=iwwv) or [Google Drive](https://drive.google.com/drive/folders/1tN9OK29SA6CgRs7jyc1ImicTxsqgDQf7?usp=sharing)|
| [EDTER-VOC-Stage II](https://github.com/MengyangPu/EDTER/#45-the-results-of-edter-voc-stage-ii-on-bsds500) | [BaiDuNetdisk](https://pan.baidu.com/s/1LmgQiCiWKrwzEuog5BQ_ng?pwd=b9rm) or [Google Drive](https://drive.google.com/drive/folders/1TXwwK-4YUa596EX6PWvYAJvtl4Hx2Wcw?usp=sharing)|

## 1 Usage
Our project is developed based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation). Please follow the official MMsegmentation [INSTALL.md](https://github.com/fudan-zvg/SETR/blob/main/docs/install.md) and [getting_started.md](https://github.com/fudan-zvg/SETR/blob/main/docs/getting_started.md) for installation and dataset preparation.

### 1.1 Linux
The full script for setting up EDTER with conda is following [SETR](https://github.com/fudan-zvg/SETR#linux).
```
conda create -n edter python=3.7 -y
conda activate edter
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch -y
pip install mmcv-full==1.2.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
cd EDTER
pip install -e .  # or "python setup.py develop"
pip install -r requirements/optional.txt
```

### 1.2 Datasets
#### BSDS500
Download the augmented BSDS500 data (1.2GB) from [HED-BSDS](https://vcl.ucsd.edu/hed/HED-BSDS.tar).<br/>
The original BSDS500 dataset can be downloaded from [Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).<br/>

```
|-- data
    |-- BSDS
        |-- ImageSets
        |   |-- train_pair.txt
        |   |-- test.txt
        |   |-- pascal_train_pair.txt
        |-- train
        |   |-- aug_data
        |   |-- aug_data_scale_0.5
        |   |-- aug_data_scale_1.5
        |   |-- aug_gt
        |   |-- aug_gt_scale_0.5
        |   |-- aug_gt_scale_1.5
        |-- test
        |   |-- 2018.jpg
        ......
```

#### PASCAL VOC
Download the augmented PASCAL VOC data from [Google Drive](https://drive.google.com/file/d/1NfL7__nVP5U0TzbF765EsC5qvt8nwtg6/view?usp=sharing) or [BaiDuNetdisk](https://pan.baidu.com/s/1d9CTR9w1MTcVrBvG-WIIXw?pwd=83cv).
```
|-- data
    |-- PASCAL
        |-- ImageSets
        |   |-- pascal_train_pair.txt
        |   |-- test.txt
        |-- aug_data
            |-- 0.0_0
            |   |-- 2008_000002.jpg
            ......
            |-- 0.0_1
            |   |-- 2008_000002.jpg
            ......
        |-- aug_gt
            |-- 0.0_0
            |   |-- 2008_000002.png
            ......
            |-- 0.0_1
            |   |-- 2008_000002. png
            ......
```

#### NYUD
Download the augmented NYUD data (~11GB) from [Google Drive](https://drive.google.com/drive/folders/1TQpKzCV4Ujkfs4V_vMasKAcvg3p4ByCN?usp=sharing) or [BaiDuNetdisk](https://pan.baidu.com/s/1J5z6235tv1xef3HXTaqnKg?pwd=t2ce).<br/>
```
|-- data
    |-- NYUD
        |-- ImageSets
        |   |-- hha-test.txt
        |   |-- hha-train.txt
        |   |-- image-test.txt
        |   |-- image-train.txt
        |-- train
            |-- GT
            |-- GT_05
            |-- GT_15
            |-- HHA
            |-- HHA_05
            |-- HHA_15
            |-- Images
            |-- Images_05
            |-- Images_15
        |-- test
            |-- HHA
            |   |-- img_5001.png
            ......
            |-- Images
            |   |-- img_5001.png
            ......
```


### 1.3 Initial weights
If you are unable to download due to network reasons, you can download the initial weights from [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth)(VIT-base-p16) and [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth)(VIT-large-p16).<br/>
The two .pth files of initial weights should be placed in the folder -- [./pretrain](https://github.com/MengyangPu/EDTER/tree/main/pretrain).
```
|-- EDTER
    |-- pretrain
        |-- jx_vit_base_p16_384-83fb41ba.pth
        |-- jx_vit_large_p16_384-b3be5167.pth
```

## 2 Training 
❗❗❗ Note: Our project only supports distributed training on multiple GPUs on one machine or a single GPU on one machine.
### 2.1 Step1: The training of EDTER-Stage I on BSDS500
If you want to set the batch size in each GPU, please refer to
https://github.com/MengyangPu/EDTER/blob/bbee219d5713a77aeec61c0f7fde93620cb02d60/configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py#L99
For example, *data = dict(samples_per_gpu=4)* means that each GPU can process 4 images.<br/>
Therefore, **the batch size of training = samples_per_gpu * GPU_NUM**. In the experiments, we set the training batch size to 8, where samples_per_gpu=4 and GPU_NUM=2.

The command to train the first-stage model is as follows
```shell
cd EDTER
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage I on the BSDS500 dataset with 2 GPUs
cd EDTER
bash ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py 2
```
### 2.2 Step2: The training of EDTER-Stage II on BSDS500
Change the '--global-model-path' in tools/train_local.py 
https://github.com/MengyangPu/EDTER/blob/ccb79b235e82ddbb4a6cc6d36c38325b674decd1/tools/train_local.py#L22-L23
```shell
cd EDTER
bash ./tools/dist_train_local.sh ${GLOBALCONFIG_FILE} ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage II on the BSDS500 dataset with 2 GPUs
cd EDTER
bash ./tools/dist_train_local.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py configs/bsds/EDTER_BIMLA_320x320_80k_bsds_local8x8_bs_8.py 2
```
### 2.3 How to train the EDTER model on BSDS-VOC (BSDS500 and PASCAL VOC Context):
### Step 1: The training of EDTER-VOC-Stage I on PASCAL VOC Context
We first pre-train Stage I on PASCAL VOC Context Dataset ([Google Drive](https://drive.google.com/file/d/1NfL7__nVP5U0TzbF765EsC5qvt8nwtg6/view?usp=sharing), [BaiDuNetdisk](https://pan.baidu.com/s/1d9CTR9w1MTcVrBvG-WIIXw?pwd=83cv)).<br/>
The command to train the first stage model on PASCAL VOC Context is as follows
```shell
cd EDTER
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage I on the PASCAL VOC Context dataset with 2 GPUs
cd EDTER
bash ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_pascal_bs_8.py 2
```
❗Note: The model trained on the PASCAL VOC Context dataset is used as the initialization model in Step 2.

### 2.4 Step 2: The training of EDTER-VOC-Stage I on BSDS500
First, we set the path of the pre-training model in [train.py](https://github.com/MengyangPu/EDTER/blob/main/tools/train.py)
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/train.py#L28-L30
For example, *parser.add_argument(
        '--load-from', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_pascal_bs_8/iter_X0000.pth ',
        help='the checkpoint file to load weights from')*

Then, we execute the following command to train the first stage model on bsds500:
```shell
cd EDTER
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage I on the BSDS500 dataset with 2 GPUs
cd EDTER
bash ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8.py 2
```
### 2.5 Step3: The training of EDTER-VOC-Stage II on BSDS500
Change the '--global-model-path' in [train_local.py](https://github.com/MengyangPu/EDTER/blob/main/tools/train_local.py).
https://github.com/MengyangPu/EDTER/blob/ccb79b235e82ddbb4a6cc6d36c38325b674decd1/tools/train_local.py#L22-L23
❗Note: According to the results in stage one, we select the best model as the global model. 
Thus, we set:
*parser.add_argument('--global-model-path', type=str, default=' ../work_dirs/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8/iter_X0000.pth',
                        help='the dir of the best global model').*

Then, the command to train the second stage model is as follows:
```shell
cd EDTER
./tools/dist_train_local.sh ${GLOBALCONFIG_FILE} ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage II on the BSDS500 dataset with 2 GPUs
cd EDTER
./tools/dist_train_local.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8.py configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_local8x8_bs_8.py 2
```

## 3 Testing
### 3.1 EDTER-Stage I with single-scale testing
First, please set the '--config', '--checkpoint', and '--tmpdir' in [test.py](https://github.com/MengyangPu/EDTER/blob/main/tools/test.py).<br/>
'--config':
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test.py#L21
'--checkpoint':
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test.py#L22
'--tmpdir':
https://github.com/MengyangPu/EDTER/blob/f060fd3c8bf1e5b1c91097721b2eafecc5f3041e/tools/test.py#L47-L50
For example:
```
#If you want to test EDTER-Stage I, please set:
parser.add_argument('--config', type=str, default='configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py', help='train config file path')
parser.add_argument('--checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_bs_8/iter_XXXXX.pth')
#If you want to test EDTER-VOC-Stage I, please set:
parser.add_argument('--config', type=str, default='configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8.py', help='train config file path')
parser.add_argument('--checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8/iter_XXXXX.pth')
```
Then, please execute the command:
```shell
cd EDTER
python ./tools/test.py
```
### 3.2 EDTER-Stage I with multi-scale testing
First, please set the '--config', '--checkpoint', and '--tmpdir' in [test.py](https://github.com/MengyangPu/EDTER/blob/main/tools/test.py).<br/>
'--config':
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test.py#L21
'--checkpoint':
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test.py#L22
'--tmpdir':
https://github.com/MengyangPu/EDTER/blob/f060fd3c8bf1e5b1c91097721b2eafecc5f3041e/tools/test.py#L47-L50
For example:
```
#If you want to test EDTER-Stage I, please set:
parser.add_argument('--config', type=str, default='configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8_ms.py', help='train config file path')
parser.add_argument('--checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_bs_8/iter_XXXXX.pth')
#If you want to test EDTER-VOC-Stage I, please set:
parser.add_argument('--config', type=str, default='configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8_ms.py', help='train config file path')
parser.add_argument('--checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8/iter_XXXXX.pth')
```
❗Note: Use the config file ending in **_ms.py** in **configs/EDTER**.

Then, please execute the command:
```shell
cd EDTER
python ./tools/test.py
```

### 3.3 EDTER-Stage II with single-scale testing
First, please set the '--globalconfig', '--config', '--global-checkpoint', '--checkpoint', and '--tmpdir' in [test_local.py](https://github.com/MengyangPu/EDTER/blob/main/tools/test_local.py).<br/>
'--globalconfig':
https://github.com/MengyangPu/EDTER/blob/84cc7355c9012a7d31cd14e25fd6c6b336714163/tools/test_local.py#L20-L21
'--config':
https://github.com/MengyangPu/EDTER/blob/84cc7355c9012a7d31cd14e25fd6c6b336714163/tools/test_local.py#L22-L23
'--checkpoint':
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test_local.py#L24-L25
'--global-checkpoint':
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test_local.py#L26-L28
'--tmpdir':
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test_local.py#L53-L56

For example:
```
#If you want to test EDTER-Stage II, please set:
parser.add_argument('--globalconfig', type=str, default='configs/bsds/**EDTER_BIMLA_320x320_80k_bsds_bs_8.py**', help='train global config file path')
parser.add_argument('--config', type=str, default='configs/bsds/**EDTER_BIMLA_320x320_80k_bsds_local8x8_bs_8.py**', help='train local config file path')
parser.add_argument('--checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_local8x8_bs_8/iter_XXXXX.pth', help='the dir of local model')
parser.add_argument('--global-checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_bs_8/iter_XXXXX.pth', help='the dir of global model')
#If you want to test EDTER-VOC-Stage II, please set:
parser.add_argument('--globalconfig', type=str, default='configs/bsds/**EDTER_BIMLA_320x320_80k_bsds_aug_bs_8.py**', help='train global config file path')
parser.add_argument('--config', type=str, default='configs/bsds/**EDTER_BIMLA_320x320_80k_bsds_aug_local8x8_bs_8.py**', help='train local config file path')
parser.add_argument('--checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_aug_local8x8_bs_8/iter_XXXXX.pth', help='the dir of local model')
parser.add_argument('--global-checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8/iter_XXXXX.pth', help='the dir of global model')
```
Please execute the command:
```shell
cd EDTER
python ./tools/test_local.py
```

### 3.4 EDTER-Stage II with multi-scale testing
First, please set the '--globalconfig', '--config', '--global-checkpoint', '--checkpoint', and '--tmpdir' in [test_local.py](https://github.com/MengyangPu/EDTER/blob/main/tools/test_local.py).<br/>
'--globalconfig':
https://github.com/MengyangPu/EDTER/blob/84cc7355c9012a7d31cd14e25fd6c6b336714163/tools/test_local.py#L20-L21
'--config':
https://github.com/MengyangPu/EDTER/blob/84cc7355c9012a7d31cd14e25fd6c6b336714163/tools/test_local.py#L22-L23
'--checkpoint':
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test_local.py#L24-L25
'--global-checkpoint':
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test_local.py#L26-L28
'--tmpdir':
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test_local.py#L53-L56

For example:
```
#If you want to test EDTER-Stage II, please set:
parser.add_argument('--globalconfig', type=str, default='configs/bsds/**EDTER_BIMLA_320x320_80k_bsds_bs_8_ms.py**', help='train global config file path')
parser.add_argument('--config', type=str, default='configs/bsds/**EDTER_BIMLA_320x320_80k_bsds_local8x8_bs_8_ms.py**', help='train local config file path')
parser.add_argument('--checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_aug_local8x8_bs_8/iter_XXXXX.pth', help='the dir of local model')
parser.add_argument('--global-checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8/iter_XXXXX.pth', help='the dir of global model')
#If you want to test EDTER-VOC-Stage II, please set:
parser.add_argument('--globalconfig', type=str, default='configs/bsds/**EDTER_BIMLA_320x320_80k_bsds_aug_bs_8_ms.py**', help='train global config file path')
parser.add_argument('--config', type=str, default='configs/bsds/**EDTER_BIMLA_320x320_80k_bsds_aug_local8x8_bs_8_ms.py**', help='train local config file path')
parser.add_argument('--checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_aug_local8x8_bs_8/iter_XXXXX.pth', help='the dir of local model')
parser.add_argument('--global-checkpoint', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8/iter_XXXXX.pth', help='the dir of global model')
```
❗Note: Use the config file ending in **_ms.py** in **configs/EDTER**.

Please execute the command:
```shell
cd EDTER
python ./tools/test_local.py
```
## 🔥🔥4 The comparison of the reported results and the reproduced results🔥🔥
### 4.1 The results of EDTER-Stage I on BSDS500
The **original results** reported in the [paper](https://arxiv.org/abs/2203.08566) (row 1 of Table 2) are as:
|   Model    | ODS  | OIS  | AP   |
| -----------| ---- | ---- | ---- |
|🔥EDTER-StageI(SS)|0.817 |0.835 |0.867 |

The **reproduced results** of EDTER-Stage I on BSDS500 are shown in the table:
|   iter   | ODS(SS)  | OIS(SS)  | AP(SS)   | ODS(MS)  | OIS(MS)  | AP(MS)   |
| ---------| ---- | ---- | ---- | ---- | ---- | ---- |
| 10k	   |0.813 |	0.830|0.861	|0.837 |0.854 | 0.890|
| 20k	   |0.816 |	0.832|0.865	|0.837 |0.853 |	0.889|
|**🔥30k(best)**   |**0.817**|**0.833**|**0.866**|**0.837**|**0.853**|**0.888**|
| 40k	   |0.815 |	0.832|0.866	|0.836 |0.853 |	0.888|
| 50k	   |0.815 |	0.832|0.866	|0.834 |0.852 |	0.887|
| 60k	   |0.813 |	0.828|0.862	|0.832 |0.849 |	0.885|
| 70k	   |0.813 |	0.829|0.864	|0.832 |0.849 |	0.884|
| 80k	   |0.813 |	0.829|0.863	|0.831 |0.849 |	0.884|

**SS: Single-Scale testing, MS: Multi-Scale testing**

**🔥All files generated during the training process, including the models and test results (.png and .mat files) for every 10k iterations, and the training logs can be downloaded through [Google Drive](https://drive.google.com/drive/folders/1vwX_gAmhCvJwbMEGGO1Hh63-xrWMzgXA?usp=sharing) or [BaiDuNetdisk](https://pan.baidu.com/s/158B9xct-J8nnOBGSPuotRA?pwd=nx35).**

### 4.2 The results of EDTER-Stage II on BSDS500
The **original results** reported in the [paper](https://arxiv.org/abs/2203.08566) (Table 3, EDTER) are as:
|    Model    | ODS(SS)  | OIS(SS)  | AP(SS)   | ODS(MS)  | OIS(MS)  | AP(MS)   |
| ------------| ---- | ---- | ---- | ---- | ---- | ---- |
|🔥EDTER-StageII|0.824 |0.841 |0.880 |0.840 |0.858 |0.896 |

The **reproduced results** of EDTER-Stage II on BSDS500 are shown in the table:
|   iter   | ODS(SS)  | OIS(SS)  | AP(SS)   | ODS(MS)  | OIS(MS)  | AP(MS)   |
| ---------| ---- | ---- | ---- | ---- | ---- | ---- |
| 10k	   |0.821 |0.838 |0.874 |0.839 |0.856 |0.893 |
| 20k	   |0.822 |0.839 |0.876 |0.838 |0.856 |0.893 |
| 30k	   |0.824 |0.841 |0.878 |0.837 |0.855 |0.893 |
|**🔥40k(best)**   |**0.825**|**0.841**|**0.880**|**0.838**|**0.855**|**0.894**|
| 50k	   |0.823 |0.840 |0.877 |0.835 |0.852 |0.892 |
| 60k	   |0.822 |0.839 |0.876 |0.834 |0.852 |0.889 |
| 70k	   |0.820 |0.837 |0.875 |0.833 |0.851 |0.890 |
| 80k	   |0.817 |0.836 |0.873 |0.829 |0.848 |0.888 |

**🔥All files generated during the training process, including the models and test results (.png and .mat files) for every 10k iterations, and the training logs can be downloaded through [Google Drive](https://drive.google.com/drive/folders/1I9C_FeV1hPM3lZzdAOkKSyeSl9OBw3Mm?usp=sharing) or [BaiDuNetdisk](https://pan.baidu.com/s/1JzlXAH8YnOEFiDncjSDZpA?pwd=mawm).**


### 4.3 The EDTER model pre-trained on the PASCAL VOC Context dataset
On the testing set of BSDS500, we report the results of **the EDTER model pre-trained on the PASCAL VOC Context dataset**, as shown in the table:
|   iter   | ODS(SS)  | OIS(SS)  | AP(SS)   |
| ---------| ---- | ---- | ---- |
| **🔥10k(best)**  |**0.775** |**0.795** |**0.835** |
| 20k      |0.767 |0.788 |0.827 |
| 30k	   |0.760 |0.777 |0.816 |
| 40k      |0.762 |0.779 |0.815 |
| 50k	   |0.755 |0.769 |0.809 |
| 60k	   |0.757 |0.771 |0.810 |
| 70k	   |0.757 |0.771 |0.810 |
| 80k	   |0.757 |0.771 |0.810 |

**🔥All files generated during the training process, including the models and test results (.png and .mat files) for every 10k iterations, and the training logs can be downloaded through [Google Drive](https://drive.google.com/drive/folders/1pFCRjHfD-Jpnxn0STOouhGM4zpof2dGe?usp=sharing) or [BaiDuNetdisk](https://pan.baidu.com/s/1SS62jBW-Qao7BQ3nXrDvYQ?pwd=dk5v).**


### 4.4 The results of EDTER-VOC-Stage I on BSDS500
The **original results** reported in the [paper](https://arxiv.org/abs/2203.08566) are **null**.

The **reproduced results** of EDTER-VOC-Stage I on BSDS500 are shown in the table:
|   iter   | ODS(SS)  | OIS(SS)  | AP(SS)   | ODS(MS)  | OIS(MS)  | AP(MS)   |
| ---------| ---- | ---- | ---- | ---- | ---- | ---- |
| 10k	   |0.823 |0.837 |0.871 |0.845 |0.861 |0.897 |
|**🔥20k(best)**   |**0.824** |**0.839** |**0.872** |**0.844** |**0.860** |**0.896** |
| 30k	   |0.822 |0.838 |0.873 |0.842 |0.858 |0.895 |
| 40k      |0.821 |0.837 |0.871 |0.842 |0.857 |0.893 |
| 50k	   |0.821 |0.836 |0.870 |0.839 |0.855 |0.891 |
| 60k	   |0.820 |0.834 |0.869 |0.840 |0.855 |0.891 |
| 70k	   |0.819 |0.835 |0.869 |0.838 |0.854 |0.890 |
| 80k	   |0.819 |0.834 |0.868 |0.838 |0.854 |0.890 |

**🔥All files generated during the training process, including the models and test results (.png and .mat files) for every 10k iterations, and the training logs can be downloaded through [Google Drive](https://drive.google.com/drive/folders/1tN9OK29SA6CgRs7jyc1ImicTxsqgDQf7?usp=sharing) or [BaiDuNetdisk](https://pan.baidu.com/s/15CIuL2r0fZckSifgNFanBw?pwd=iwwv).**

### 4.5 The results of EDTER-VOC-Stage II on BSDS500
The **original results** reported in the [paper](https://arxiv.org/abs/2203.08566) (Table 3, EDTER-VOC) are as:
|    Model    | ODS(SS)  | OIS(SS)  | AP(SS)   | ODS(MS)  | OIS(MS)  | AP(MS)   |
| ------------| ---- | ---- | ---- | ---- | ---- | ---- |
|🔥EDTER-VOC-Stage II|0.832 |0.847 |0.886 |0.848 |0.865 |0.903 |

The **reproduced results** of EDTER-VOC-Stage II on BSDS500 are shown in the table:
|   iter   | ODS(SS)  | OIS(SS)  | AP(SS)   | ODS(MS)  | OIS(MS)  | AP(MS)   |
| ---------| ---- | ---- | ---- | ---- | ---- | ---- |
| 10k	   |0.827 |0.844 |0.880 |0.846 |0.861 |0.900 |
|**🔥20k(best)**   |**0.829** |**0.845** |**0.883** |**0.846** |**0.862** |**0.901** |
| 30k	   |0.829 |0.845 |0.883 |0.843 |0.860 |0.899 |
| 40k      |0.826 |0.842 |0.882 |0.841 |0.858 |0.897 |
| 50k	   |0.823 |0.838 |0.878 |0.837 |0.854 |0.893 |
| 60k	   |0.821 |0.837 |0.878 |0.834 |0.852 |0.892 |
| 70k	   |0.816 |0.833 |0.872 |0.831 |0.848 |0.888 |
| 80k	   |0.815 |0.832 |0.871 |0.830 |0.848 |0.887 |

**🔥All files generated during the training process, including the models and test results (.png and .mat files) for every 10k iterations, and the training logs can be downloaded through [Google Drive](https://drive.google.com/drive/folders/1TXwwK-4YUa596EX6PWvYAJvtl4Hx2Wcw?usp=sharing) or [BaiDuNetdisk](https://pan.baidu.com/s/1LmgQiCiWKrwzEuog5BQ_ng?pwd=b9rm).**


## 5 [Eval](https://github.com/MengyangPu/EDTER/tree/main/eval)
#### BSDS500
```shell
cd eval
run eval_bsds.m
```
#### NYUD
Download the matfile (NYUD) from [Google Drive](https://drive.google.com/file/d/1MFSQUl9G5ETynowU0EgAosdb5BprEfaU/view?usp=sharing) or [BaiDuNetdisk](https://pan.baidu.com/s/1LEeoOKkzdcewmPhq5z86wA?pwd=25p8).<br/>
```shell
cd eval
run eval_nyud.m
```

## 6 Results
If you want to compare your method with EDTER, you can download the pre-computed results:<br/>
BSDS500: [Google Drive](https://drive.google.com/file/d/1zL74whvVnrZAe-j2BveLD1yZrsrk-Vb5/view?usp=sharing). <br/>
NYUD: [Google Drive](https://drive.google.com/drive/folders/19HAfNIQB7sJ83Vj-7qgK5irOocAUtiN1?usp=sharing) or [BaiDuNetdisk](https://pan.baidu.com/s/1xy5JOqs_zLpOoTOlzb5Bxw?pwd=b941).

## 7 Download Pre-trained model

| model                                            | Pre-trained Model                                                              |
| ------------------------------------------------ | ------------------------------------------------------------------------------ | 
|[EDTER-BSDS-VOC-StageI](configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8.py/)           | [BaiDuNetdisk](https://pan.baidu.com/s/1xxkDRUoy9vfO6rtjx_GOqA?pwd=l282) or [Google Drive](https://drive.google.com/drive/folders/1OkdakKKIMRGnKH8mxuFi_qI9sa903CD2?usp=share_link)|
|[EDTER-BSDS-VOC-StageII](configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_local8x8_bs_8.py/)          | [BaiDuNetdisk](https://pan.baidu.com/s/1hT1v24H6GjjjjMXqe_Knuw?pwd=skjw) or [Google Drive](https://drive.google.com/drive/folders/1OkdakKKIMRGnKH8mxuFi_qI9sa903CD2?usp=share_link)|
|[EDTER-NYUD-RGB-StageI](configs/nyud/EDTER_BIMLA_320x320_40k_nyud_rgb_bs_4.py/)          | [BaiDuNetdisk](https://pan.baidu.com/s/1nkBuXV1s4YUpkmu-7eYV-w?pwd=dwdi) or [Google Drive](https://drive.google.com/file/d/1D88fULxXrPXXp-NrsB2RNMpA8aCzvldN/view?usp=sharing)|
|[EDTER-NYUD-RGB-StageII](configs/nyud/EDTER_BIMLA_320x320_40k_nyud_rgb_local8x8_bs_4.py/)          | [BaiDuNetdisk](https://pan.baidu.com/s/1ihUbPeVr5cRt_vef4pkBZQ?pwd=s00u) or [Google Drive](https://drive.google.com/file/d/1XhBfexCZaBqBsaoWza-w6moahwULa6gm/view?usp=sharing)|
|[EDTER-NYUD-HHA-StageI](configs/nyud/EDTER_BIMLA_320x320_40k_nyud_hha_bs_4.py/)          | [BaiDuNetdisk](https://pan.baidu.com/s/1xzPela1UYTNa9Mdk-i_G-A?pwd=ko2f) or [Google Drive](https://drive.google.com/file/d/1AOXgR7Ulw_4eFPZVi2seT0J3wG9rdC9U/view?usp=sharing)|
|[EDTER-NYUD-HHA-StageII](configs/nyud/EDTER_BIMLA_320x320_40k_nyud_hha_local8x8_bs_4.py/)          | [BaiDuNetdisk](https://pan.baidu.com/s/1huMD4Ecop6ACrK1O4VToNA?pwd=p7wu) or [Google Drive](https://drive.google.com/file/d/15qYXelMjMWH4r9_f3tHL5V4M3PbGjQFD/view?usp=sharing)|

## ❗❗❗Important notes
- ❗❗❗All the models are trained and tested on a single machine with multiple NVIDIA-V100-32G GPUs.
- ❗❗❗Training on distributed GPUs is not supported.

## Acknowledgements
- We thank the anonymous reviewers for their valuable and inspiring comments and suggestions.
- Thanks to the previous open-sourced repo:<br/>
  [SETR](https://github.com/fudan-zvg/SETR)<br/>
  [MMsegmentation](https://github.com/open-mmlab/mmsegmentation)<br/>

## Reference
```bibtex
@InProceedings{Pu_2022_CVPR,
    author    = {Pu, Mengyang and Huang, Yaping and Liu, Yuming and Guan, Qingji and Ling, Haibin},
    title     = {EDTER: Edge Detection With Transformer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {1402-1412}
}
```
