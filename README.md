# EDTER
> [EDTER: Edge Detection with Transformer](https://arxiv.org/abs/2203.08566)                 
> Mengyang Pu, Yaping Huang, Yuming Liu, Qingji Guan and Haibin Ling                 
> *CVPR 2022*

Please refer to [supplementary material](https://github.com/MengyangPu/EDTER/blob/main/supp/EDTER-supp.pdf) for more results.

## Usage

### Datasets
#### BSDS500
Download the augmented BSDS500 data (1.2GB) from [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz).<br/>
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
#### NYUD
Download the augmented NYUD data (~11GB) from [here](https://pan.baidu.com/s/1J5z6235tv1xef3HXTaqnKg)(Code:t2ce).<br/>
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

### Pre-trained model
If you are unable to download due to network reasons, you can download the pre-trained model from [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth) and [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth).

### Training
#### The training of Stage I
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage I on BSDS500 dataset with 8 GPUs
./tools/dist_train.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py 8
```
#### The training of Stage II
Change the '--global-model-path' in [train_local.py](https://github.com/MengyangPu/EDTER/blob/main/tools/train_local.py).
```shell
./tools/dist_train_local.sh ${GLOBALCONFIG_FILE} ${CONFIG_FILE} ${GPU_NUM} 
# For example, train Stage II on BSDS500 dataset with 8 GPUs
./tools/dist_train_local.sh configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py configs/bsds/EDTER_BIMLA_320x320_80k_bsds_local8x8_bs_8.py 8
```

### Testing
#### Single-scale testing
Change the '--config', '--checkpoint', and '--tmpdir' in [test.py](https://github.com/MengyangPu/EDTER/blob/main/tools/test.py).
```shell
python tools/test.py
```

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
