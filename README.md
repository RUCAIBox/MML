# Code for MML

This is the resource code for our work.
> Xingyu Pan, Yushuo Chen, Changxin Tian, Zihan Lin, Jinpeng Wang, He Hu and Wayne Xin Zhao. "Multimodal Meta-Learning for Cold Start Sequential Recommendation"

## Overview
We purpose a Multimodal Meta-Learning (denoted as MML) method to introduce multimodal side information of items (e.g., text and image) into the meta-learning process to stably improve the recommendation performance for cold-start users. Specifically, we model a unique sequence for each kind of multimodal information, and purpose a multimodel meta-learner framework to distill the global knowledge from the multimodal information. Meanwhile, we design a cold-start item embedding generator, which apply the multimodal information to warm up the ID embedding of new items. 

<p align="center">
  <img src="model_fig.png" alt="MML structure" width="600">
  <br>
  <b>Figure</b>: Overall Architecture of MML
</p>

## Reproducing
We provide the script for reproducing the experimental results in this repository.
For example, if you want to get the result for `hangzhou`, 
you should firstly download the data files and put them into `dataset/hangzhou` folder.

**NOTE: Due to privacy policies, our data is not availiable temporarily. We will make data masking and release the desensitized data after camera ready.**

And then you can execute the following command:
```bash
bash script/run_meta.sh hangzhou 0
```
`0` represent the GPU id to execute our code, you can change it as you need. 

We also provide the script for showing the result in this repository.
You can execute the following command to get the result for our model.
```bash
bash script/print_result.sh
```

## Acknowledgement
Our implementation is based on [RecBole](https://github.com/RUCAIBox/RecBole) framework.
