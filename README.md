# Effective Model Sparsification by Scheduled Grow-and-Prune Methods
ICLR 2022 paper "[Effective Model Sparsification by Scheduled Grow-and-Prune Methods](https://openreview.net/pdf?id=xa6otUDdP2W)". [Model](https://drive.google.com/drive/folders/1-bY4Bbu1zF5Y7VzPZYBaZyyG3QQ9glTz?usp=sharing) and test [code](https://drive.google.com/drive/folders/103t6gqmqB2_LoMrj-9-DVllLDzUxtOu0?usp=sharing) are available for downloading.

Please see an example of GaP with Transformer.


## Computer Vision

Model [Download](https://drive.google.com/drive/folders/12WFSp9rmcfFZL7JWucHU6g7IEbLJoi19?usp=sharing)

| Models  | Method | Partition | Sparsity Ratio | Sparsity Distribution | Top-1 Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| ResNet-50 | S-GaP | 4 | 0.8 | Uniform | 77.856% |
| ResNet-50 | S-GaP | 4 | 0.8 | Non-uniform | 78.132% |
| ResNet-50 | P-GaP | 4 | 0.8 | Uniform | 77.492% |
| ResNet-50 | S-GaP | 4 | 0.9 | Uniform | 76.348% |
| ResNet-50 | S-GaP | 4 | 0.9 | Non-uniform | 77.896% |
| ResNet-50 | P-GaP | 4 | 0.9 | Uniform | 76.128% |


## Machine Translation (WMT-14 EN-DE) 

Model [Download](https://drive.google.com/drive/folders/1do0GrxPwg9ghRaYXLQ7LxKpX4hqD0_jh?usp=sharing)

| Models  | Method | Partition | Sparsity Ratio | Sparsity Distribution | BLEU Score |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Transformer | S-GaP | 3 | 0.8 | Uniform | 27.59 |
| Transformer | S-GaP | 6 | 0.8 | Uniform | 27.65 |
| Transformer | P-GaP | 3 | 0.8 | Uniform | 27.93 |
| Transformer | P-GaP | 6 | 0.8 | Uniform | 27.67 |
| Transformer | S-GaP | 3 | 0.9 | Uniform | 27.72 |
| Transformer | S-GaP | 6 | 0.9 | Uniform | 27.06 |
| Transformer | P-GaP | 3 | 0.9 | Uniform | 27.31 |
| Transformer | P-GaP | 6 | 0.9 | Uniform | 26.88 |


## 3D Object Part Segmentation with PointNet++ on ShapeNet

Model [Download](https://drive.google.com/drive/folders/1UyMBbUoihLmd1yfjVB6O9rHqgS1akzZ4?usp=sharing)

## Object Detection (SSD on COCO-2017) 

Model [Download](https://drive.google.com/drive/folders/1L9VQnSKQ2n58gWpwja3zy_ac3KMVv_BW?usp=sharing)

## Citation
if you find this repo is helpful, please cite
```
@inproceedings{ma2022effective,
    title={Effective Model Sparsification by Scheduled Grow-and-Prune Methods},
    author={Xiaolong Ma and Minghai Qin and Fei Sun and Zejiang Hou and Kun Yuan and Yi Xu and Yanzhi Wang and Yen-Kuang Chen and Rong Jin and Yuan Xie},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=xa6otUDdP2W}
}
```


