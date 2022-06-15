# EDTER
> [EDTER: Edge Detection with Transformer](https://arxiv.org/abs/2203.08566)                 
> Mengyang Pu, Yaping Huang, Yuming Liu, Qingji Guan and Haibin Ling                 
> *CVPR 2022*

Please refer to [supplementary material](https://github.com/MengyangPu/EDTER/blob/main/supp/EDTER-supp.pdf) for more results.

## Usage

### Datasets
#### BSDS500
Download the augmented BSDS500 data (1.2GB) from [here](http://vcl.ucsd.edu/hed/HED-BSDS.tar).<br/>
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
python test.py
```

#### Multi-scale testing
Change the '--globalconfig', '--config', '--global-checkpoint', '--checkpoint', and '--tmpdir' in [test_local.py](https://github.com/MengyangPu/EDTER/blob/main/tools/test_local.py).<br/>
Use the config file ending in _ms.py in configs/EDTER.
```shell
python test_local.py
```

### Results
If you want to compare your method with EDTER, you can download the precomputed results [here](https://drive.google.com/file/d/1zL74whvVnrZAe-j2BveLD1yZrsrk-Vb5/view?usp=sharing).

### Download Pre-trained model.

| model                                            | Pre-trained Model                                                              |
| ------------------------------------------------ | ------------------------------------------------------------------------------ | 
|[BSDS-StageI]()               | [Coming Soon]()  |
|[BSDS-StageII]()              | [Coming Soon]()  |
|[BSDS-Aug-StageI]()           | [Coming Soon]()  |
|[BSDS-Aug-StageII]()          | [Coming Soon]()  |

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
