# Img2Vid-DYAN

## Purpose and Introduction
This project presents a way of performing video generation, by integrating parts of [Seg2Vid](https://github.com/junting/seg2vid) into [DYAN](https://github.com/liuem607/DYAN). Authors Pan et. al. present the novel task of video generation from a single image in their paper _Video Generation from Single Semantic Label Map_. Lui et al. present DYAN, a neural network framework for video generation using an optical flow sequence as input. _DYAN_ has been shown to produce good results for video generation, but is limited by the need to use an computationally expensive optical flow generator, such as PyFlow, to generate optical flow inputs. 

In this project, I present a new method for obtaining optical flow inputs for _DYAN_, using portions of the _Seg2Vid_ network as a neural network optical flow generator. By merging these two networks, I have shown better video generation than the _Seg2Vid_ network on its own for the _PlayingViolin_ portion of the UCF-101 dataset. Results can be explored in the ablation study.

## Architecture
_Seg2Vid_ is used to generate optical flow inputs for _DYAN_. These optical flows are then manually scaled up, because the optical flows generated by _Seg2Vid_ are much smaller than _DYAN_ 's optical flow inputs.

![Text](https://github.com/pat-hanbury/Img2Vid-DYAN/blob/master/FlowChart.png)

## Acknowledgements
Many thanks to Wen Lui and Professor Octavia Camps of Northeastern University for guidance with this project.

## References
[1] [Video Generation from Single Semantic Label Map, Pan et. al.](https://github.com/junting/seg2vid)

[2] [DYAN: A Dynamical Atoms-Based Network for Video Prediction, Lui et. al.](https://github.com/liuem607/DYAN)

[3] [Learned Perceptual Image Patch Similarity (LPIPS) metric, Zhang et. al.](https://github.com/richzhang/PerceptualSimilarity)


