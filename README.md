# StyleSwin

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleswin-transformer-based-gan-for-high-1/image-generation-on-celeba-hq-1024x1024)](https://paperswithcode.com/sota/image-generation-on-celeba-hq-1024x1024?p=styleswin-transformer-based-gan-for-high-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleswin-transformer-based-gan-for-high-1/image-generation-on-celeba-hq-256x256)](https://paperswithcode.com/sota/image-generation-on-celeba-hq-256x256?p=styleswin-transformer-based-gan-for-high-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleswin-transformer-based-gan-for-high-1/image-generation-on-ffhq-256-x-256)](https://paperswithcode.com/sota/image-generation-on-ffhq-256-x-256?p=styleswin-transformer-based-gan-for-high-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleswin-transformer-based-gan-for-high-1/image-generation-on-lsun-churches-256-x-256)](https://paperswithcode.com/sota/image-generation-on-lsun-churches-256-x-256?p=styleswin-transformer-based-gan-for-high-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/styleswin-transformer-based-gan-for-high-1/image-generation-on-ffhq)](https://paperswithcode.com/sota/image-generation-on-ffhq?p=styleswin-transformer-based-gan-for-high-1)

![Teaser](imgs/teaser.png)

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen), [Shuyang Gu](http://home.ustc.edu.cn/~gsy777/), [Bo Zhang](https://bo-zhang.me/), [Jianmin Bao](https://jianminbao.github.io/), [Dong Chen](http://www.dongchen.pro/), [Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/), [Yong Wang](https://auto.ustc.edu.cn/2021/0510/c25976a484888/page.htm) and [Baining Guo](microsoft.com/en-us/research/people/bainguo/).

This repo is the official implementation of "[StyleSwin: Transformer-based GAN for High-resolution Image Generation]()".

Code and pretrained models will be availiable soon.

## Abstract

> Despite the tantalizing success in a broad of vision tasks, transformers have not yet demonstrated on-par ability as ConvNets in high-resolution image generative modeling. In this paper, we seek to explore using pure transformers to build a generative adversarial network for high-resolution image synthesis. To this end, we believe that local attention is crucial to strike the balance between computational efficiency and modeling capacity. Hence, the proposed generator adopts Swin transformer in a style-based architecture. To achieve a larger receptive field, we propose double attention which simultaneously leverages the context of the local and the shifted windows, leading to improved generation quality. Moreover, we show that offering the knowledge of the absolute position that has been lost in window-based transformers greatly benefits the generation quality. The proposed StyleSwin is scalable to high resolutions, with both the coarse geometry and fine structures benefit from the strong expressivity of transformers. However, blocking artifacts occur during high-resolution synthesis because performing the local attention in a block-wise manner may break the spatial coherency. To solve this, we empirically investigate various solutions, among which we find that employing a wavelet discriminator to examine the spectral discrepancy effectively suppresses the artifacts. Extensive experiments show the superiority over prior transformer-based GANs, especially on high resolutions, e.g., $1024\times 1024$. The StyleSwin, without complex training strategies, excels over StyleGAN on CelebA-HQ $1024$, and achieves on-par performance on FFHQ-$1024$, proving the promise of using transformers for high-resolution image generation.

## Main Results

### Quantitative Results

| Dataset | Resolution | FID | Pretrained Model |
| :-: | :-: | :-: | :-: |
| FFHQ | 256x256 | 2.81 | [Google Drive]()/[Azure Storage](https://facevcstandard.blob.core.windows.net/v-bowenz/output/styleswin_final_results/FFHQ256/FFHQ_256.pt?sv=2020-08-04&st=2021-12-30T03%3A25%3A24Z&se=2121-12-31T03%3A25%3A00Z&sr=b&sp=r&sig=0IZQPXTeSvwctJozTf4HokJ2CLLndZbWEViDBmFh7Jo%3D) |
| LSUN Church | 256x256 | 2.95 | [Google Drive]()/[Azure Storage](https://facevcstandard.blob.core.windows.net/v-bowenz/output/styleswin_final_results/LSUNChurch256/LSUNChurch_256.pt?sv=2020-08-04&st=2021-12-30T03%3A28%3A37Z&se=2121-12-31T03%3A28%3A00Z&sr=b&sp=r&sig=c6qo6p5uUEsRljIaJaDCJNZZLp4hVXB2GCoP8Zmpj%2Bw%3D) |
| CelebA-HQ | 256x256 | 3.25 | [Google Drive]()/[Azure Storage](https://facevcstandard.blob.core.windows.net/v-bowenz/output/styleswin_final_results/CelebAHQ256/CelebAHQ_256.pt?sv=2020-08-04&st=2021-12-30T03%3A30%3A58Z&se=2121-12-31T03%3A30%3A00Z&sr=b&sp=r&sig=T0FSCAkM6Vr6kxTErjxK5D3AvpdFb%2FWQzalH6AOz4TA%3D) |
| FFHQ | 1024x1024 | 5.07 | [Google Drive]()/[Azure Storage](https://facevcstandard.blob.core.windows.net/v-bowenz/output/styleswin_final_results/FFHQ1024/FFHQ_1024.pt?sv=2020-08-04&st=2021-12-30T03%3A32%3A33Z&se=2121-12-31T03%3A32%3A00Z&sr=b&sp=r&sig=lyAJbzZH9PCi5Pl1xJUAP0vxcPAKMxhlxt%2F6FbKbrYw%3D) |
| CelebA-HQ | 1024x1024 | 4.43 | [Google Drive]()/[Azure Storage](https://facevcstandard.blob.core.windows.net/v-bowenz/output/styleswin_final_results/CelebAHQ1024/CelebAHQ_1024.pt?sv=2020-08-04&st=2021-12-30T03%3A33%3A34Z&se=2121-12-31T03%3A33%3A00Z&sr=b&sp=r&sig=pLZ%2B1vKftaJM1eUJnjtmVTpHMrPhpQllT5Ms0s2OrGQ%3D) |

### Qualitative Results

Image samples of FFHQ-1024 generated by StyleSwin:

![](imgs/ffhq.jpg)

Image samples of CelebA-HQ 1024 generated by StyleSwin:

![](imgs/celebahq.jpg)

Latent code interpolation examples of FFHQ-1024 between the left-most and the right-most images:

![](imgs/latent_interpolation.jpg)

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

## Acknowledgements

This code borrows heavily from [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) and [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). We also thank the contributors of code [Positional Encoding in GANs](https://github.com/open-mmlab/mmgeneration/blob/master/configs/positional_encoding_in_gans/README.md), [DiffAug](https://github.com/mit-han-lab/data-efficient-gans) and [StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).

## Maintenance

This project is currently maintained by Bowen Zhang. If you have any questions, feel free to contact [zhangbowen@mail.ustc.edu.cn](zhangbowen@mail.ustc.edu.cn).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
