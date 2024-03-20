# IFNet: An Image-Enhanced Cross-Modal Fusion Network for Radiology Report Generation

## Abstract

Radiology Report Generation (RRG) aims to automatically generate descriptive textual reports for medical images through computer-assisted technologies, which can alleviate the workload of radiologists, reduce the probability of misdiagnoses, and mitigate the strain on medical resources. However, previous studies explore rarely improving low-quality images in datasets, integrating cross-modal information, and optimizing network latency. To advance the field of RRG and address its existing challenges, we propose an Image-enhanced cross-modal Fusion Network (IFNet) for the automated generation of radiological reports. IFNet comprises three meticulously designed modules: an image enhancement module for augmenting the granular abnormal structural representations of X-ray images; a cross-modal fusion network capable of capturing the interactive relations of cross-modal features comprehensively and efficiently; and a Transformer report generation module with linear time complexity, aimed at efficiently producing radiological reports with reduced network latency and operability on resource-constrained devices. Experiments on the public dataset IU-Xray demonstrate significant achievements of IFNet, surpassing the performance of the current state-of-the-art methods.

![architecture](.\architecture.png)

## Requirements

python = 3.10.9

torch: 1.11.0+cu113

torchvision = 0.12.0

scikit-learn = 1.3.0

opencv-python = 4.8.0.76

numpy = 1.25.0

pycocoevalcap = 1.2

## Datasets

You can download the `IU-Xray` dataset from https://openi.nlm.nih.gov/faq and then put the files in `data/iu_xray`.

## Image Enhancement Module

Run `modules/IEC.py` to enhance images in the dataset.

## Text Encoder

Extract global text features through Chexbert and place the extracted labels in `data\iu_xray`.

You can get Chexbert from https://github.com/stanfordmlgroup/CheXbert

## Experiments on IU-Xray

Run `bash run_iu_xray.sh` to train a model on the IU-Xray data.