# StyleSwin

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleswin-transformer-based-gan-for-high-1/image-generation-on-celeba-hq-1024x1024)](https://paperswithcode.com/sota/image-generation-on-celeba-hq-1024x1024?p=styleswin-transformer-based-gan-for-high-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleswin-transformer-based-gan-for-high-1/image-generation-on-celeba-hq-256x256)](https://paperswithcode.com/sota/image-generation-on-celeba-hq-256x256?p=styleswin-transformer-based-gan-for-high-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleswin-transformer-based-gan-for-high-1/image-generation-on-ffhq-256-x-256)](https://paperswithcode.com/sota/image-generation-on-ffhq-256-x-256?p=styleswin-transformer-based-gan-for-high-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleswin-transformer-based-gan-for-high-1/image-generation-on-lsun-churches-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-churches-256-x-256?p=styleswin-transformer-based-gan-for-high-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleswin-transformer-based-gan-for-high-1/image-generation-on-ffhq-1024-x-1024)](https://paperswithcode.com/sota/image-generation-on-ffhq-1024-x-1024?p=styleswin-transformer-based-gan-for-high-1)
[![WebDemo](https://img.shields.io/badge/%F0%9F%A4%97%20Web%20Demo-Huggingface-blue)](https://huggingface.co/spaces/hysts/StyleSwin)

![Teaser](imgs/teaser.png)

This repo is the official implementation of "[StyleSwin: Transformer-based GAN for High-resolution Image Generation](https://arxiv.org/abs/2112.10762)" (CVPR 2022).

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen), [Shuyang Gu](http://home.ustc.edu.cn/~gsy777/), [Bo Zhang](https://bo-zhang.me/), [Jianmin Bao](https://jianminbao.github.io/), [Dong Chen](http://www.dongchen.pro/), [Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/), [Yong Wang](https://auto.ustc.edu.cn/2021/0510/c25976a484888/page.htm) and [Baining Guo](microsoft.com/en-us/research/people/bainguo/).

## Abstract

> Despite the tantalizing success in a broad of vision tasks, transformers have not yet demonstrated on-par ability as ConvNets in high-resolution image generative modeling. In this paper, we seek to explore using pure transformers to build a generative adversarial network for high-resolution image synthesis. To this end, we believe that local attention is crucial to strike the balance between computational efficiency and modeling capacity. Hence, the proposed generator adopts Swin transformer in a style-based architecture. To achieve a larger receptive field, we propose double attention which simultaneously leverages the context of the local and the shifted windows, leading to improved generation quality. Moreover, we show that offering the knowledge of the absolute position that has been lost in window-based transformers greatly benefits the generation quality. The proposed StyleSwin is scalable to high resolutions, with both the coarse geometry and fine structures benefit from the strong expressivity of transformers. However, blocking artifacts occur during high-resolution synthesis because performing the local attention in a block-wise manner may break the spatial coherency. To solve this, we empirically investigate various solutions, among which we find that employing a wavelet discriminator to examine the spectral discrepancy effectively suppresses the artifacts. Extensive experiments show the superiority over prior transformer-based GANs, especially on high resolutions, e.g., 1024x1024. The StyleSwin, without complex training strategies, excels over StyleGAN on CelebA-HQ 1024x1024, and achieves on-par performance on FFHQ 1024x1024, proving the promise of using transformers for high-resolution image generation.

## Quantitative Results

| Dataset | Resolution | FID | Pretrained Model |
| :-: | :-: | :-: | :-: |
| FFHQ | 256x256 | 2.81 | [Google Drive](https://drive.google.com/file/d/1OjYZ1zEWGNdiv0RFKv7KhXRmYko72LjO/view?usp=sharing)/[Azure Storage](https://facevcstandard.blob.core.windows.net/v-bowenz/output/styleswin_final_results/FFHQ256/FFHQ_256.pt?sv=2020-10-02&st=2022-03-14T12%3A36%3A35Z&se=2099-12-31T15%3A59%3A00Z&sr=b&sp=r&sig=QBETIToFQ8MtlnnVLpNlbcPB8MPTZkiDDTNjlgovf%2Fo%3D) |
| LSUN Church | 256x256 | 2.95 | [Google Drive](https://drive.google.com/file/d/1HF0wFNuz1WFrqGEbPhOXjL4QrY05Zu_m/view?usp=sharing)/[Azure Storage](https://facevcstandard.blob.core.windows.net/v-bowenz/output/styleswin_final_results/LSUNChurch256/LSUNChurch_256.pt?sv=2020-10-02&st=2022-03-14T12%3A37%3A41Z&se=2099-12-31T15%3A59%3A00Z&sr=b&sp=r&sig=VPWMsvHbJKUj8v6a9gp2u424OAS9o%2BL1qKKfGtYWMN8%3D) |
| CelebA-HQ | 256x256 | 3.25 | [Google Drive](https://drive.google.com/file/d/1YtIJOgLFfkaMI_KL2gBQNABFb1cwOzvM/view?usp=sharing)/[Azure Storage](https://facevcstandard.blob.core.windows.net/v-bowenz/output/styleswin_final_results/CelebAHQ256/CelebAHQ_256.pt?sv=2020-10-02&st=2022-03-14T12%3A39%3A42Z&se=2099-12-31T15%3A59%3A00Z&sr=b&sp=r&sig=xasn7w5ou739bM9NAwmA3HEFkxKXOrqddH76EviXewo%3D) |
| FFHQ | 1024x1024 | 5.07 | [Google Drive](https://drive.google.com/file/d/17-ILwzLBoHq4HTdAPeaCug7iBvxKWkvp/view?usp=sharing)/[Azure Storage](https://facevcstandard.blob.core.windows.net/v-bowenz/output/styleswin_final_results/FFHQ1024/FFHQ_1024.pt?sv=2020-10-02&st=2022-03-14T12%3A40%3A20Z&se=2099-12-31T15%3A59%3A00Z&sr=b&sp=r&sig=Di7J57LLvayVVTmEymjI61y42q%2BxS9pxCmBHbay6t%2Bk%3D) |
| CelebA-HQ | 1024x1024 | 4.43 | [Google Drive](https://drive.google.com/file/d/1y3wkykjvCbteTaGTRF8EedkG-N1Z8jFf/view?usp=sharing)/[Azure Storage](https://facevcstandard.blob.core.windows.net/v-bowenz/output/styleswin_final_results/CelebAHQ1024/CelebAHQ_1024.pt?sv=2020-10-02&st=2022-03-14T12%3A40%3A49Z&se=2099-12-31T15%3A59%3A00Z&sr=b&sp=r&sig=gicenvMBClfUmFr1gew06exUoN033JWADmNLCCtyu4w%3D) |

## Requirements

To install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Generating image samples with pretrained model

To generate 50k image samples of resolution **1024** and evaluate the fid score:

```bash
python -m torch.distributed.launch --nproc_per_node=1 train_styleswin.py --sample_path /path_to_save_generated_samples --size 1024 --ckpt /path/to/checkpoint --eval --val_num_batches 12500 --val_batch_size 4 --eval_gt_path /path_to_real_images_50k
```

To generate 50k image samples of resolution **256** and evaluate the fid score:

```bash
python -m torch.distributed.launch --nproc_per_node=1 train_styleswin.py --sample_path /path_to_save_generated_samples --size 256 --G_channel_multiplier 2 --ckpt /path/to/checkpoint --eval --val_num_batches 12500 --val_batch_size 4 --eval_gt_path /path_to_real_images_50k
```

## Training

### Data preparing

When training FFHQ and CelebA-HQ, we use `ImageFolder` datasets. The data structure is like this:

```
FFHQ
├── images
│  ├── 000001.png
│  ├── ...
```

When training LSUN Church, please follow [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch#usage) to create a lmdb dataset first. After this, the data structure is like this:

```
LSUN Church
├── data.mdb
└── lock.mdb
```

### FFHQ-1024

To train a new model of **FFHQ-1024** from scratch:

```bash
python -m torch.distributed.launch --nproc_per_node=8 train_styleswin.py --batch 2 --path /path_to_ffhq_1024 --checkpoint_path /tmp --sample_path /tmp --size 1024 --D_lr 0.0002 --D_sn --ttur --eval_gt_path /path_to_ffhq_real_images_50k --lr_decay --lr_decay_start_steps 600000
```

### CelebA-HQ 1024

To train a new model of **CelebA-HQ 1024** from scratch:

```bash
python -m torch.distributed.launch --nproc_per_node=8 train_styleswin.py --batch 2 --path /path_to_celebahq_1024 --checkpoint_path /tmp --sample_path /tmp --size 1024 --D_lr 0.0002 --D_sn --ttur --eval_gt_path /path_to_celebahq_real_images_50k
```

### FFHQ-256

To train a new model of **FFHQ-256** from scratch:

```bash
python -m torch.distributed.launch --nproc_per_node=8 train_styleswin.py --batch 4 --path /path_to_ffhq_256 --checkpoint_path /tmp --sample_path /tmp --size 256 --G_channel_multiplier 2 --bcr --D_lr 0.0002 --D_sn --ttur --eval_gt_path /path_to_ffhq_real_images_50k --lr_decay --lr_decay_start_steps 775000 --iter 1000000
```

### CelebA-HQ 256

To train a new model of **CelebA-HQ 256** from scratch:

```bash
python -m torch.distributed.launch --nproc_per_node=8 train_styleswin.py --batch 4 --path /path_to_celebahq_256 --checkpoint_path /tmp --sample_path /tmp --size 256 --G_channel_multiplier 2 --bcr --r1 5 --D_lr 0.0002 --D_sn --ttur --eval_gt_path /path_to_celebahq_real_images_50k --lr_decay --lr_decay_start_steps 500000
```

### LSUN Church 256

To train a new model of **LSUN Church 256** from scratch:

```bash
python -m torch.distributed.launch --nproc_per_node=8 train_styleswin.py --batch 4 --path /path_to_lsun_church_256 --checkpoint_path /tmp --sample_path /tmp --size 256 --G_channel_multiplier 2 --use_flip --r1 5 --lmdb --D_lr 0.0002 --D_sn --ttur --eval_gt_path /path_to_lsun_church_real_images_50k --lr_decay --lr_decay_start_steps 1300000 --iter 1500000
```

**Notice**: When training on 16 GB GPUs, you could add `--use_checkpoint` to save GPU memory.

## Qualitative Results

Image samples of FFHQ-1024 generated by StyleSwin:

![](imgs/ffhq.jpg)

Image samples of CelebA-HQ 1024 generated by StyleSwin:

![](imgs/celebahq.jpg)

Latent code interpolation examples of FFHQ-1024 between the left-most and the right-most images:

![](imgs/latent_interpolation.jpg)

## Citing StyleSwin

```
@misc{zhang2021styleswin,
      title={StyleSwin: Transformer-based GAN for High-resolution Image Generation}, 
      author={Bowen Zhang and Shuyang Gu and Bo Zhang and Jianmin Bao and Dong Chen and Fang Wen and Yong Wang and Baining Guo},
      year={2021},
      eprint={2112.10762},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Responsible AI Considerations

Our work does not directly modify the exiting images which may alter the identity or expression of the people. We discourage the use of our work in such applications as it is not designed to do so. We have quantitatively verified that the proposed method does not show evident disparity, on gender and ages as the model mostly follows the dataset distribution; however, we encourage additional care if you intend to use the system on certain demographic groups. We also encourage use of fair and representative data when training on customized data. We caution that the high-resolution images produced by our model may potentially be misused for impersonating humans and viable solutions so avoid this include adding tags or watermarks when distributing the generated photos.

## Acknowledgements

This code borrows heavily from [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). We also thank the contributors of code [Positional Encoding in GANs](https://github.com/open-mmlab/mmgeneration/blob/master/configs/positional_encoding_in_gans/README.md), [DiffAug](https://github.com/mit-han-lab/data-efficient-gans), [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) and [GIQA](https://github.com/cientgu/GIQA).

## Maintenance

This is the codebase for our research work. Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact [zhangbowen@mail.ustc.edu.cn](zhangbowen@mail.ustc.edu.cn) or [zhanbo@microsoft.com](zhanbo@microsoft.com).


## License
The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
