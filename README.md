# TOQ-Nets-PyTorch-Release
Pytorch implementation for the Temporal and Object Quantification Networks (TOQ-Nets).

![TOQ-Nets](http://toqnet.csail.mit.edu/data/img/model.png)

**[Temporal and Object Quantification Networks](http://toqnet.csail.mit.edu/data/papers/2021IJCAI-TOQNet.pdf)**
<br />
[Jiayuan Mao](http://jiayuanm.com), 
[Zhezheng Luo](https://superurop.mit.edu/scholars/zhezheng-luo/),
[Chuang Gan](http://people.csail.mit.edu/ganchuang/), 
[Joshua B. Tenenbaum](https://web.mit.edu/cocosci/josh.html), 
[Jiajun Wu](https://jiajunwu.com/),
[Leslie Pack Kaelbling](https://people.csail.mit.edu/lpk/), and
[Tomer D. Ullman](https://www.tomerullman.org/)
<br />
In International Joint Conference on Artificial Intelligence (IJCAI) 2021 (Poster)
<br />
[[Paper]](http://toqnet.csail.mit.edu/data/papers/2021IJCAI-TOQNet.pdf)
[[Project Page]](http://toqnet.csail.mit.edu/)
[[BibTex]](http://toqnet.csail.mit.edu/data/bibtex/2021IJCAI-TOQNet.bib)

```latex
@inproceedings{Mao2021Temporal,
    title={{Temporal and Object Quantification Networks}},
    author={Mao, Jiayuan and Luo, Zhezheng and Gan, Chuang and Tenenbaum, Joshua B. and Wu, Jiajun and Kaelbling, Leslie Pack and Ullman, Tomer D.},
    booktitle={International Joint Conferences on Artificial Intelligence},
    year={2021}
}
```

## Prerequisites

- Python 3
- PyTorch 1.0 or higher, with NVIDIA CUDA Support
- Other required python packages specified by `requirements.txt`. See the Installation.

## Installation

Install [Jacinle](https://github.com/vacancy/Jacinle): Clone the package, and add the bin path to your global `PATH` environment variable:

```bash
git clone https://github.com/vacancy/Jacinle --recursive
export PATH=<path_to_jacinle>/bin:$PATH
```

Clone this repository:

```bash
git clone https://github.com/vacancy/TOQ-Nets-PyTorch --recursive
```

Create a conda environment for TOQ-Nets, and install the requirements. This includes the required python packages
from both Jacinle TOQ-Nets. Most of the required packages have been included in the built-in `anaconda` package:

```bash
conda create -n nscl anaconda
conda install pytorch torchvision -c pytorch
```

## Dataset preparation

We evaluate our model on four datasets: Soccer Event, RLBench, Toyota Smarthome and Volleyball. To run the experiments, you need to prepare them under `NSPCL-Pytorch/data`.

#### Soccer Event
[Download link](http://toqnet.csail.mit.edu/data/datasets/gfootball.zip)

#### RLBenck
[Download link](http://toqnet.csail.mit.edu/data/datasets/rlbench.zip)

#### Toyota Smarthome
Dataset can be obtained from the website: [Toyota Smarthome: Real-World Activities of Daily Living](https://project.inria.fr/toyotasmarthome/)
```latex
@InProceedings{Das_2019_ICCV,
    author = {Das, Srijan and Dai, Rui and Koperski, Michal and Minciullo, Luca and Garattoni, Lorenzo and Bremond, Francois and Francesca, Gianpiero},
    title = {Toyota Smarthome: Real-World Activities of Daily Living},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

#### Volleyball
Dataset can be downloaded from this [github repo](https://github.com/mostafa-saad/deep-activity-rec#dataset).
```latex
@inproceedings{msibrahiCVPR16deepactivity,
  author    = {Mostafa S. Ibrahim and Srikanth Muralidharan and Zhiwei Deng and Arash Vahdat and Greg Mori},
  title     = {A Hierarchical Deep Temporal Model for Group Activity Recognition.},
  booktitle = {2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}
```

## Training and evaluation.

#### Standard 9-way classification task

To train the model on the standard 9-way classification task on the soccer dataset:

```bash
jac-crun <gpu_ids> scripts/action_classification_softmax.py -t 1001 --run_name 9_way_classification -Mmodel-name "'NLTL_SAv3'" -Mdata-name "'LongVideoNvN'" -Mn_epochs 200 -Mbatch_size 128 -Mhp-train-estimate_inequality_parameters "(1,1)" -Mmodel-both_quantify False -Mmodel-depth 0
```
The hyper parameter `estimate_inequality_parameters` is to estimate the distribution of input physical features, and is only required when training TOQ-Nets (but not for baselines). 

#### Few-shot actions

To train on regular actions and test on new actions:

```bash
jac-crun <gpu_ids> scripts/action_classification_softmax.py  -t 1002 --run_name few_shot -Mdata-name "'TrajectorySingleActionNvN_Wrapper_FewShot_Softmax'" -Mmodel-name "'NLTL_SAv3'" -Mlr 3e-3 -Mn_epochs 200 -Mbatch_size 128 -Mdata-new_actions "[('interfere', (50, 50, 2000)), ('sliding', (50, 50, 2000))]" -Mhp-train-finetune_period "(1,200)" -Mhp-train-estimate_inequality_parameters "(1,1)"
```

You can set the split of few-shot actions using `-Mdata-new_actions`, and the tuple `(50, 50, 2000)` represents the number of samples available in training validation and testing.

#### Generalization to more of fewer players and  temporally warped trajectories.

To test the generalization to more or fewer players, as well as temporal warpped trajectories, first train the model on the standard 6v6 games:

```bash
jac-crun <gpu_ids> scripts/action_classification_softmax.py -t 1003 --run_name generalization -Mmodel-name "'NLTL_SAv3'" -Mdata-name "'LongVideoNvN'" -Mdata-n_players 6 -Mn_epochs 200 -Mbatch_size 128 -Mhp-train-estimate_inequality_parameters "(1,1)" -Mlr 3e-3
```

Then to generalize to games with 11 players:

```bash
jac-crun 3 scripts/action_classification_softmax.py -t 1003 --run_name generalization_more_players --eval 200 -Mdata-name "'LongVideoNvN'" -Mdata-n_train 0.1 -Mdata-temporal "'exact'" -Mdata-n_players 11
```

The number `200` after `--eval` should be equal to the number of epochs of training. Note that `11` can be replace by any number of players from `[3,4,6,8,11]`.

Similarly, to generalize to temporally warped trajectoryes:

```bash
jac-crun 3 scripts/action_classification_softmax.py -t 1003 --run_name generalization_time_warp --eval 200 -Mdata-name "'LongVideoNvN'" -Mdata-n_train 0.1 -Mdata-temporal "'all'" -Mdata-n_players 6
```

#### Baselines

We also provide the example commands for training all baselines:

**STGCN**

```bash
jac-crun <gpu_ids> scripts/action_classification_softmax.py -t 1004 --run_name stgcn -Mmodel-name "'STGCN_SA'" -Mdata-name "'LongVideoNvN'" -Mdata-n_players 6 -Mmodel-n_agents 13 -Mn_epochs 200 -Mbatch_size 128
```

**STGCN-LSTM**

```bash
jac-crun <gpu_ids> scripts/action_classification_softmax.py -t 1005 --run_name stgcn_lstm -Mmodel-name "'STGCN_LSTM_SA'" -Mdata-name "'LongVideoNvN'" -Mdata-n_players 6 -Mmodel-n_agents 13 -Mn_epochs 200 -Mbatch_size 128
```

**Space-Time Region Graph**

```bash
jac-crun <gpu_ids> scripts/action_classification_softmax.py -t 1006 --run_name strg -Mmodel-name "'STRG_SA'" -Mdata-name "'LongVideoNvN'" -Mn_epochs 200 -Mbatch_size 128
```

**Non-Local**

```bash
jac-crun <gpu_ids> scripts/action_classification_softmax.py -t 1007 --run_name non_local -Mmodel-name "'NONLOCAL_SA'" -Mdata-name "'LongVideoNvN'" -Mn_epochs 200 -Mbatch_size 128
```

