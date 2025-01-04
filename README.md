# HARG
source code for the paper HARG: Hierarchical Adaptive Reasoning Graph for Activity Parsing

## Requirements
This code was tested with following environment:
> PyTorch == 1.7.0
> 
> torchvision == 0.8.0
> 
> cudatoolkit == 11.0

## Dataset
Download the MOMA dataset and the MOMA-LRG dataset from the official website using this [link](https://moma.stanford.edu).

## Dataset Prepare
### MOMA Dataset:
> cd tools
> 
> python MOMA_create_datasets.py

### MOMA-LRG Dataset:
> python MOMA_LRG_create_datasets.py

## Usage
### Training
> python main.py

### Validation
> python main.py --mode val

## Model Zoo

| Dataset  | Atomic action mAP | Sub-activity acc@1 | Sub-activity acc@5 | Sub-activity mAP | Activity acc@1 | Activity acc@5 | Activity mAP |                                                         Models                                                          |
|:--------:|:-----------------:|:------------------:|:------------------:|:----------------:|:--------------:|:--------------:|:------------:|:-----------------------------------------------------------------------------------------------------------------------:|
|   MOMA   |       40.23       |       73.57        |       97.89        |      84.08       |     95.35      |     99.79      |    99.03     |                            [link](https://pan.baidu.com/s/1iA9BBgDAbXfxxFu-vp6hIg?pwd=0000)                             |
| MOMA-LRG |       46.76       |       65.32        |       96.61        |      71.55       |     96.01      |     99.25      |    99.00     |                            [link](https://pan.baidu.com/s/1VTrhQ0-rlq7PUzaA-6QGQg?pwd=0000)                             |

## Code components
```
HARG
├── tools
|    ├── MOMA_anns
|    |   ├── split_by_trim
|    |   |      ├── train.txt
|    |   |      └── val.txt     
|    |   ├──  graph_anns.json
|    |   ├ ...
|    ├── MOMA_LRG_anns
|    |   ├── splits
|    |   |      ├── few_shot.json
|    |   |      └── standard.json     
|    |   ├──  anns.json
|    |   ├ ...
|    ├── MOMA_create_datasets.py
|    ├── MOMA_tools.py
|    ├── MOMA_LRG_create_datasets.py
|    ├── MOMA_LRG_tools.py
|    ├ ...
├── dataloader
|    ├── iterator_factory.py
|    ├── video_iterator.py
├ ...
├── data
|    ├── MOMA
|    |   ├── trim_videos
|    |   ├── all_frames
|    |   ├── video_level_object
|    |   ├── video_level_relation_A_O
|    |   ├── dataset_level_train.txt
|    |   ├── dataset_level_val.txt
|    |   ├── adjacent
|    ├ ...
|    ├── MOMA_LRG
|    |   ├── interaction
|    |   ├── extract_frames
|    |   ├── video_level_object
|    |   ├── video_level_relation_A_O
|    |   ├── dataset_level_train.txt
|    |   ├── dataset_level_val.txt
|    |   ├── adjacent
|    ├ ...
├ ...
├── scripts
|   ├── train_AMS_R50_sthv1_rgb_8f.sh
|   ├──train_AMS_R50_sthv1_rgb_16f
|   ├ ... 
├ ...
```

## Baselines

We provide the following baselines for comparison: VideoMAEv2, AMS-Net, MAR and OR2G. The code of these baselines can be found in the [baselines](https://github.com/whuoyj/HARG/blob/main/baselines/README.md) folder.

## Acknowledgments

We sincerely appreciate the work and the accompanying code of [VideoMAEv2](https://github.com/open-mmlab/mmaction2), [AMS-Net](https://github.com/open-mmlab/mmaction2), [MAR](https://github.com/alibaba-mmai-research/Masked-Action-Recognition), and the [MMAction2](https://github.com/open-mmlab/mmaction2) framework toolbox. We are grateful for the hard research work and dedication of the authors.
