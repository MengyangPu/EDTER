# EDTER
> [EDTER: Edge Detection with Transformer](https://arxiv.org/abs/2203.08566)                 
> Mengyang Pu, Yaping Huang, Yuming Liu, Qingji Guan and Haibin Ling                 
> *CVPR 2022*

Please refer to [supplementary material](https://github.com/MengyangPu/EDTER/blob/main/supp/EDTER-supp.pdf) for more results.

## Usage

Our project is developed based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation). Please follow the official MMsegmentation [INSTALL.md](https://github.com/fudan-zvg/SETR/blob/main/docs/install.md) and [getting_started.md](https://github.com/fudan-zvg/SETR/blob/main/docs/getting_started.md) for installation and dataset preparation.

### Linux
The full script for setting up EDTER with conda is following [here](https://github.com/fudan-zvg/SETR).
```
conda create -n edter python=3.7 -y
conda activate edter
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch -y
pip install mmcv-full==1.2.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
cd EDTER
pip install -e .  # or "python setup.py develop"
pip install -r requirements/optional.txt
```

### Datasets
#### BSDS500
Download the augmented BSDS500 data (1.2GB) from [here](https://vcl.ucsd.edu/hed/HED-BSDS.tar).<br/>
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
Download the augmented PASCAL VOC data from [here](https://pan.baidu.com/s/1d9CTR9w1MTcVrBvG-WIIXw?pwd=83cv)
(Code:83cv).

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
Download the augmented NYUD data (~11GB) from [here](https://pan.baidu.com/s/1J5z6235tv1xef3HXTaqnKg?pwd=t2ce)(Code:t2ce).<br/>
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


### initial weights
If you are unable to download due to network reasons, you can download the initial weights from [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth)(VIT-base-p16) and [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth)(VIT-large-p16).
The two .pth files of initial weights should be placed in the [folder](https://github.com/MengyangPu/EDTER/tree/main/pretrain).
```
|-- EDTER
    |-- pretrain
        |-- jx_vit_base_p16_384-83fb41ba.pth
        |-- jx_vit_large_p16_384-b3be5167.pth
```

### Training 
Note: Our project only supports distributed training on multiple GPUs on one machine or a single GPU on one machine.
#### Step1: The training of Stage I on BSDS500
If you want to set the batch size in each GPU, please refer to
https://github.com/MengyangPu/EDTER/blob/bbee219d5713a77aeec61c0f7fde93620cb02d60/configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py#L99
For example, data = dict(samples_per_gpu=4) means that each GPU can process 4 images.
Therefore, the batch size of training = samples_per_gpu * GPU_NUM. In the experiments, we set the training batch size to 8, where samples_per_gpu=4 and GPU_NUM=2.

The command to train the first-stage model is as follows
```shell
cd EDTER
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage I on the BSDS500 dataset with 2 GPUs
cd EDTER
bash ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py 2
```
#### Step2: The training of Stage II on BSDS500
Change the '--global-model-path' in tools/train_local.py 
https://github.com/MengyangPu/EDTER/blob/ccb79b235e82ddbb4a6cc6d36c38325b674decd1/tools/train_local.py#L22-L23
```shell
cd EDTER
bash ./tools/dist_train_local.sh ${GLOBALCONFIG_FILE} ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage II on the BSDS500 dataset with 2 GPUs
cd EDTER
bash ./tools/dist_train_local.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py configs/bsds/EDTER_BIMLA_320x320_80k_bsds_local8x8_bs_8.py 2
```
### How to train the EDTER model on BSDS-VOC (BSDS500 and PASCAL VOC Context)
#### Step 1: The training of Stage I on PASCAL VOC Context
We first pre-train Stage I on [PASCAL VOC Context Dataset](https://pan.baidu.com/s/1d9CTR9w1MTcVrBvG-WIIXw?pwd=83cv)
The command to train the first stage model on PASCAL VOC Context is as follows
```shell
cd EDTER
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage I on the PASCAL VOC Context dataset with 2 GPUs
cd EDTER
bash ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_pascal_bs_8.py 2
```
Note: The model trained on the PASCAL VOC Context dataset is used as the initialization model in Step2.

#### Step 2: The training of Stage I on BSDS500
First, we set the path of the pre-training model in [train.py](https://github.com/MengyangPu/EDTER/blob/main/tools/train.py)
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/train.py#L28-L30
For example, parser.add_argument(
        '--load-from', type=str, default='../work_dirs/EDTER_BIMLA_320x320_80k_pascal_bs_8/iter_X0000.pth ',
        help='the checkpoint file to load weights from')

Then, we execute the following command to train the first stage model on bsds500:
```shell
cd EDTER
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage I on the BSDS500 dataset with 2 GPUs
cd EDTER
bash ./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8.py 2
```
#### Step3: The training of Stage II on BSDS500
Change the '--global-model-path' in [train_local.py](https://github.com/MengyangPu/EDTER/blob/main/tools/train_local.py).
https://github.com/MengyangPu/EDTER/blob/ccb79b235e82ddbb4a6cc6d36c38325b674decd1/tools/train_local.py#L22-L23
Note: According to the results in stage one, we select the best model as the global model. 
Thus, we set:
parser.add_argument('--global-model-path', type=str, default=' ../work_dirs/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8/iter_X0000.pth',
                        help='the dir of the best global model').

Then, the command to train the second stage model is as follows:
```shell
cd EDTER
./tools/dist_train_local.sh ${GLOBALCONFIG_FILE} ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage II on the BSDS500 dataset with 2 GPUs
cd EDTER
./tools/dist_train_local.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8.py configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_local8x8_bs_8.py 2
```

### Testing
#### Stage I with single-scale testing
First, please set the '--config', '--checkpoint', and '--tmpdir' in [test.py](https://github.com/MengyangPu/EDTER/blob/main/tools/test.py).
For example:
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test.py#L21
https://github.com/MengyangPu/EDTER/blob/3b1751abec5f0add6849393a9cbf2a8e73cc65f5/tools/test.py#L22
https://github.com/MengyangPu/EDTER/blob/f060fd3c8bf1e5b1c91097721b2eafecc5f3041e/tools/test.py#L47-L50
Then, please execute the command:
```shell
cd EDTER
python ./tools/test.py
```

The original results reported in the [paper](https://arxiv.org/abs/2203.08566) (row 1 of Table 2) are as:
ODS=0.817, OIS=0.835, AP=0.867


|   iter   | ODS  | OIS  | AP   | ODS  | OIS  | AP   |
| ---------| ---- | ---- | ---- | ---- | ---- | ---- |
| 10k	   |0.813 |	0.830|0.861	|0.837 |0.854 | 0.890|
| 20k	   |0.816 |	0.832|0.865	|0.837 |0.853 |	0.889|
|**30k**   |**0.817**|**0.833**|**0.866**|**0.837**|**0.853**|**0.888**|
| 40k	   |0.815 |	0.832|0.866	|0.836 |0.853 |	0.888|
| 50k	   |0.815 |	0.832|0.866	|0.834 |0.852 |	0.887|
| 60k	   |0.813 |	0.828|0.862	|0.832 |0.849 |	0.885|
| 70k	   |0.813 |	0.829|0.864	|0.832 |0.849 |	0.884|
| 80k	   |0.813 |	0.829|0.863	|0.831 |0.849 |	0.884|

All files generated during the training process, including the models and test results (.png and .mat files) for every 10k iterations, and the training logs can be downloaded through this [BaiDuNetdisk](https://pan.baidu.com/s/158B9xct-J8nnOBGSPuotRA?pwd=nx35).


#### Multi-scale testing
Change the '--globalconfig', '--config', '--global-checkpoint', '--checkpoint', and '--tmpdir' in [test_local.py](https://github.com/MengyangPu/EDTER/blob/main/tools/test_local.py).<br/>
Use the config file ending in _ms.py in configs/EDTER.
```shell
python tools/test_local.py
```
### [Eval](https://github.com/MengyangPu/EDTER/tree/main/eval)
#### BSDS500
```shell
cd eval
run eval_bsds.m
```
#### NYUD
Download the matfile(NYUD) from [here](https://pan.baidu.com/s/1LEeoOKkzdcewmPhq5z86wA)(Code:25p8).<br/>
```shell
cd eval
run eval_nyud.m
```

### Results
If you want to compare your method with EDTER, you can download the precomputed results [BSDS500](https://drive.google.com/file/d/1zL74whvVnrZAe-j2BveLD1yZrsrk-Vb5/view?usp=sharing) and [NYUD](https://pan.baidu.com/s/1xy5JOqs_zLpOoTOlzb5Bxw)(code:b941).

### Download Pre-trained model.

| model                                            | Pre-trained Model                                                              |
| ------------------------------------------------ | ------------------------------------------------------------------------------ | 
|[EDTER-BSDS-VOC-StageI](configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_bs_8.py/)           | [BaiDuNetdisk](https://pan.baidu.com/s/1xxkDRUoy9vfO6rtjx_GOqA)  (Code:l282) or [Google Drive](https://drive.google.com/drive/folders/1OkdakKKIMRGnKH8mxuFi_qI9sa903CD2?usp=share_link)|
|[EDTER-BSDS-VOC-StageII](configs/bsds/EDTER_BIMLA_320x320_80k_bsds_aug_local8x8_bs_8.py/)          | [BaiDuNetdisk](https://pan.baidu.com/s/1hT1v24H6GjjjjMXqe_Knuw)  (Code:skjw) or [Google Drive](https://drive.google.com/drive/folders/1OkdakKKIMRGnKH8mxuFi_qI9sa903CD2?usp=share_link)|
|[EDTER-NYUD-RGB-StageI](configs/nyud/EDTER_BIMLA_320x320_40k_nyud_rgb_bs_4.py/)          | [BaiDuNetdisk](https://pan.baidu.com/s/1nkBuXV1s4YUpkmu-7eYV-w)  (Code:dwdi)  |
|[EDTER-NYUD-RGB-StageII](configs/nyud/EDTER_BIMLA_320x320_40k_nyud_rgb_local8x8_bs_4.py/)          | [BaiDuNetdisk](https://pan.baidu.com/s/1ihUbPeVr5cRt_vef4pkBZQ)  (Code:s00u)  |
|[EDTER-NYUD-HHA-StageI](configs/nyud/EDTER_BIMLA_320x320_40k_nyud_hha_bs_4.py/)          | [BaiDuNetdisk](https://pan.baidu.com/s/1xzPela1UYTNa9Mdk-i_G-A)  (Code:ko2f)  |
|[EDTER-NYUD-HHA-StageII](configs/nyud/EDTER_BIMLA_320x320_40k_nyud_hha_local8x8_bs_4.py/)          | [BaiDuNetdisk](https://pan.baidu.com/s/1huMD4Ecop6ACrK1O4VToNA)  (Code:p7wu)  |

### Important notes
- All the models are trained and tested on a single machine with multiple NVIDIA-V100-32G GPUs.
- Training on distributed GPUs is not supported.

## Acknowledgments
- We thank the anonymous reviewers for valuable and inspiring comments and suggestions.
- Thanks to previous open-sourced repo:<br/>
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
