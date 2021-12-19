# StyleSwin

![Teaser](imgs/teaser.png)

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen), [Shuyang Gu](http://home.ustc.edu.cn/~gsy777/), [Bo Zhang](https://bo-zhang.me/), [Jianmin Bao](https://jianminbao.github.io/), [Dong Chen](http://www.dongchen.pro/), [Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/), [Yong Wang](https://auto.ustc.edu.cn/2021/0510/c25976a484888/page.htm) and [Baining Guo](microsoft.com/en-us/research/people/bainguo/).

This repo is the official implementation of "[StyleSwin: Transformer-based GAN for High-resolution Image Generation]()".

## Abstract

> Despite the tantalizing success in a broad of vision tasks, transformers have not yet demonstrated on-par ability as ConvNets in high-resolution image generative modeling. In this paper, we seek to explore using pure transformers to build a generative adversarial network for high-resolution image synthesis. To this end, we believe that local attention is crucial to strike the balance between computational efficiency and modeling capacity. Hence, the proposed generator adopts Swin transformer in a style-based architecture. To achieve a larger receptive field, we propose double attention which simultaneously leverages the context of the local and the shifted windows, leading to improved generation quality. Moreover, we show that offering the knowledge of the absolute position that has been lost in window-based transformers greatly benefits the generation quality. The proposed StyleSwin is scalable to high resolutions, with both the coarse geometry and fine structures benefit from the strong expressivity of transformers. However, blocking artifacts occur during high-resolution synthesis because performing the local attention in a block-wise manner may break the spatial coherency. To solve this, we empirically investigate various solutions, among which we find that employing a wavelet discriminator to examine the spectral discrepancy effectively suppresses the artifacts. Extensive experiments show the superiority over prior transformer-based GANs, especially on high resolutions, \eg, $1024\times 1024$. The StyleSwin, without complex training strategies, excels over StyleGAN on CelebA-HQ $1024$, and achieves on-par performance on FFHQ-$1024$, proving the promise of using transformers for high-resolution image generation.

## Main Results

| Dataset | Resolution | FID | Pretrained Model |
| :-: | :-: | :-: | :-: |
| FFHQ | 256x256 | 2.81 | - |
| LSUN Church | 256x256 | 3.17 | - |
| CelebA-HQ | 256x256 | 3.25 | - |
| FFHQ | 1024x1024 | 5.07 | - |
| CelebA-HQ | 1024x1024 | 4.43 | - |

Pretrained models will be availiable soon.

## Requirements

To install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Generating image samples with pretrained model

To generate 50k image samples and evaluate the fid score:

```bash
python -m torch.distributed.launch --nproc_per_node=1 train_styleswin.py --sample_path /path_to_save_generated_samples --size 1024 --ckpt /path/to/checkpoint --eval --val_num_batches 12500 --val_val_batch_size 4 --eval_gt_path /path_to_real_images_50k
```

## Training

### FFHQ-1024

To train a new model of FFHQ-1024 from scratch:

```bash
python -m torch.distributed.launch --nproc_per_node=8 train_styleswin.py --batch 2 /path_to_ffhq_1024 --checkpoint_path /tmp --sample_path /tmp --size 1024 --D_lr 0.0002 --D_sn --ttur --eval_gt_path /path_to_ffhq_real_images_50k --lr_decay --lr_decay_start_steps 600000
```

## Citation

TBA.

## Acknowledgements

This code borrows heavily from [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch). We also thank Rui Xu for his code of [Positional Encoding in GANs](https://github.com/open-mmlab/mmgeneration/blob/master/configs/positional_encoding_in_gans/README.md) and Shengyu Zhao for his code of [DiffAug](https://github.com/mit-han-lab/data-efficient-gans).

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
