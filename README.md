# VaLiD_LVLMs_hallucinations
<div align="center">
Yifei Gao<sup>1</sup>, Jiaqi Wang<sup>1</sup> and, Jitao Sang*<sup>1,2</sup>
</div>
<div align="center">
<sup>1</sup>Beijing Jiaotong University    <sup>2</sup>Peng cheng Laboratory
</div>
<div align="center">
*Corresponding Author
</div>

<!-- Arxiv Link, Project Link -->
<div style='display:flex; gap: 0.25rem; '>
<a href="https://arxiv.org/abs/2411.15839"><img src="https://img.shields.io/badge/arXiv-2403.08542-b31b1b.svg"></a>
<a href="https://github.com/LucusFigoGao/AIGCs_Confuse_AI_Too"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-blue.svg'></a>
</div>

## Introduction
Despite the prosperity of generative models, the risks and challenges posed by AIGC cannot be overlooked. This paper pioneers an exploration into the impact of synthetic images on hallucination problems during the reasoning process of Large Vision Language Models. Extensive experimental results have confirmed a significant deviation between synthetic image- and natural image-induced hallucination, referring to as the hallucination bias.

## Update
- [9/1] Source code of ImageTranslation will be released soon.
- [9/1] We release the synthetic image dataset FakeAMBER [LINK](https://pan.baidu.com/s/1WfqVhXqeQudMFMMrWA4r1w?pwd=d54y).
- [7/16] 🎉🎉🎉 Our work is **accepted** to ACMMM 2024！
- [3/8] AIGCs_Confuse_AI_too Paper online [LINK](https://arxiv.org/abs/2403.08542).

## Image Download
* Download the images from this [LINK](https://pan.baidu.com/s/1WfqVhXqeQudMFMMrWA4r1w?pwd=d54y).
* The annotation information is same as [AMBER](https://github.com/junyangwang0410/AMBER/tree/master).

## Overview
<img src="./assets/semantic translation_01.jpg" alt="image-20230517233229650" style="zoom:80%;" />
We establish an evaluation environment for hallucinations involving synthetic images. Specifically, we introduce an automated annotation method (Semantic Translation) for unlabeled synthetic image datasets, where the generation process of synthetic images remains unchanged. This approach ensures the authenticity and consistency of the synthetic images, providing a foundation for quantitative analysis in subsequent hallucination evaluations. 

## Results
<img src="./assets/AMBER_eval.png" style="zoom:80%;" />
<img src="./assets/Redar.jpg" style="zoom:80%;" />

The main findings regarding hallucinations induced by synthetic images include (1) **a greater quantity** and (2) **a more uniform position distribution of hallucinated contents**. 

## Examples
<img src="./assets/Introduction.jpg" style="zoom:80%;" />

## Acknowledgments
This codebase borrows from most notably [AMBER](https://github.com/junyangwang0410/AMBER/tree/master) and [POPE](https://github.com/RUCAIBox/POPE). Many thanks to the authors for generously sharing their codes!


## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:

```
@inproceedings{
gao2024aigcs,
title={{AIGC}s Confuse {AI} Too: Investigating and Explaining Synthetic Image-induced Hallucinations in Large Vision-Language Models},
author={Yifei Gao and Jiaqi Wang and Zhiyu Lin and Jitao Sang},
booktitle={ACM Multimedia 2024},
year={2024},
url={https://openreview.net/forum?id=5qOWYC2IQj}
}
```
