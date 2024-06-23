# WebCeph2k Dataset and implementation code
## A New Benchmark and Low Computational Cost Localization Method for Cephalometric Analysis

PyTorch implementation for cephalometric landmark detection on ISBI2015 and WebCeph2k datasets.

## Abstract
In this study, we present WebCeph2k, an extensive and diverse cephalometric landmark localization dataset that surpasses previous benchmark datasets in terms of number of landmark annotations. This diverse cephalometric landmarks dataset has significant value in medical imaging research. Existing studies predominantly focus on datasets obtained from a single medical center and provider, which offers a limited number of landmarks and a limited diversity of cephalograms, resulting in models that exhibit low robustness and generalization when applied to more diverse datasets. The clinical application of cephalometry is hampered by significant localization errors in landmark localization models, in addition to the inadequacy of existing datasets' landmarks for clinical cephalometric diagnosis. The limited generalization ability and the occurrence of "overfitting" in deep learning models are mainly caused by the small size in the dataset. In the medical field, the inclusion of large and diverse datasets can greatly improve the generalization and performance of landmark localization models. This paper presents our WebCeph2k dataset from 9 medical centers, covering 9 different imaging devices, which surpasses the only publicly available ISBI2015 dataset in terms of sample size and number of landmarks. In addition, this study employs a low computational cost methodology to achieve optimal landmarks localization: 1) ROI regions of X-ray images are derived by exploiting the prior distribution of the data, 2) the model computational cost is reduced by adopting a spatial-depth transformation strategy, 3) the standard heatmap decoding method is optimized by integrating a compensation strategy.  The results show that the proposed method not only achieves competitive localization results to other state-of-the-art approaches, but also offers a reduction of the model computational cost, resulting in faster inference. Consequently, this research offers valuable prospects in the field of general-purpose medical landmark localization methods. We also find that our proposed dataset is more complex and challenging than the ISBI dataset.

## Prerequisites
- Ubuntu 18.04
- NVIDIA GPU 
- python3.8+

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/switch626/WebCeph2k
```

- Install python packages
```bash
pip3 install -r requirements.txt
```

### Preparing Datasets
Download the dataset from [BaiduYun][1](Password:wc2k). (Update CSV file in dataset)

### Usage

Prepare the WebCeph2k dataset or the ISBI2015 dataset.

```
python GetROI_before_train.py
```

### Train
```
python tools/train.py --cfg experinmets/ceph/training_testing.yaml
```

### Test
```
python tools/test.py --cfg experinmets/ceph/training_testing.yaml --model-file ./output/training_testing/BestModel.pth
```

## Acknowledgments
Great thanks for these papers and their open-source codes：[HR-Net](https://github.com/HRNet), [AAOF Craniofacial Growth Legacy Collection](https://aaoflegacycollection.org/aaof_home.html).

[1]:https://pan.baidu.com/s/1kQMeFARM-hh1bF-lUU7u3A?pwd=wc2k
