
# ManiSkill-Vitac Challenge 2025
<br>

[Click here to view our challenge webpage.](https://ai-workshops.github.io/maniskill-vitac-challenge-2025/)

## **Table of Contents**

- [Update](#update)
- [Installation](#installation)
- [Track list](#Track_list)
  - [Track 1](#Track-1)
  - [Track 2](#Track-2)
  - [Track 3](#Track-3)
- [Leaderboard](#leaderboard)
- [Real Robot Evaluation](#real-robot-evaluation)
- [Contact](#contact)
- [Citation](#citation)

## Update
**2024/10/22** Track 3 is online and 

---
## Installation

**Requirements:**

- Python 3.8.x-3.11.x
- GCC 7.2 upwards (Linux)
- CUDA Toolkit 11.8 or higher
- Git LFS installed (https://git-lfs.github.com/)


Clone this repo with

```bash
git clone https://github.com/cyliizyz/ManiSkill-ViTac2025.git
```

Run

```bash
conda env create -f environment.yaml
conda activate mani_vitac
```

Then install [SapienIPC](https://github.com/Rabbit-Hu/sapienipc-exp), following the [README](https://github.com/Rabbit-Hu/sapienipc-exp/blob/main/README.md) file in that repo.
When installing SAPIEN, please ensure that the installed version is from https://github.com/haosulab/SAPIEN/releases/tag/3.0.0b1 to avoid potential compatibility issues.
---
## Track_list

### Track 1

For track 1, you can modify the network structure in the [solutions](Track_1%2Fsolutions) and save your model in the [policies.py](Track_1%2Fsolutions%2Fpolicies.py). 

Please note: you are not allowed to modify the contents of [peg_insertion_sim_evaluation.py](Track_1%2Fscripts%2Fpeg_insertion_sim_evaluation.py)
and [open_lock_sim_evaluation.py](Track_1%2Fscripts%2Fopen_lock_sim_evaluation.py). We will check the MD5 hash of them.



#### Training Example

To train our example policy, run

```bash
# example policy for peg insertion
python Track_1/scripts/universal_training_script.py --cfg configs/parameters/peg_insertion.yaml
# example policy for open lock
python Track_1/scripts/universal_training_script.py --cfg configs/parameters/long_open_lock.yaml
```
#### Submission 
For policy evaluation in simulation, run

```bash
# evaluation of peg insertion and lock opening
python Track_1/scripts/peg_insertion_sim_evaluation.py --team_name [your_teamname] --model_name [your_model_name] --policy_file_path [your_best_model_path]
python Track_1/scripts/open_lock_sim_evaluation.py --team_name [your_teamname] --model_name [your_model_name] --policy_file_path [your_best_model_path]
```
Submit the evaluation logs by emailing them to [maniskill.vitac@gmail.com](maniskill.vitac@gmail.com)


---

### Track 2
Coming soon.


### Track 3

For track 3, you can modify the network structure in the [solutions](Track_1%2Fsolutions) and save your model in the [policies.py](Track_1%2Fsolutions%2Fpolicies.py). 

Please note: you are not allowed to modify the contents of [peg_insertion_sim_evaluation.py](Track_1%2Fscripts%2Fpeg_insertion_sim_evaluation.py)
and [open_lock_sim_evaluation.py](Track_1%2Fscripts%2Fopen_lock_sim_evaluation.py). We will check the MD5 hash of them.

This section focuses on optimizing the sensor's structure based on GelSight Mini. You can use modeling software to create a model of the silicone part and then import it into our simulation environment for verification.
For more specific methods, please refer to our documentation.

#### Training Example

To train our example policy, run

```bash
# example policy for peg insertion
python Track_1/scripts/universal_training_script.py --cfg configs/parameters/peg_insertion.yaml
# example policy for open lock
python Track_1/scripts/universal_training_script.py --cfg configs/parameters/long_open_lock.yaml
```

#### Submission 
For policy evaluation in simulation, run

```bash
# evaluation of peg insertion and lock opening
python Track_1/scripts/peg_insertion_sim_evaluation.py --team_name [your_teamname] --model_name [your_model_name] --policy_file_path [your_best_model_path]
python Track_1/scripts/open_lock_sim_evaluation.py --team_name [your_teamname] --model_name [your_model_name] --policy_file_path [your_best_model_path]
```
Submit the evaluation logs by emailing them to [maniskill.vitac@gmail.com](maniskill.vitac@gmail.com)



---
## Leaderboard

The leaderboard for this challenge is available at [*_Leader board_*](https://ai-workshops.github.io/maniskill-vitac-challenge-2025/#leaderboard).

---
## Real Robot Evaluation
Real robot evaluation code demo is contained in `real_env_demo/`. The GelsightMini sensor code is maintained at [GitHub/gelsight_mini_ros](https://github.com/RVSATHU/gelsight_mini_ros/).

---
## Contact

Join our [discord](https://discord.gg/CKucPQxQPr) to contact us. You may also email us at [maniskill.vitac@gmail.com](maniskill.vitac@gmail.com)

---
## Citation

```
@ARTICLE{chen2024tactilesim2real,
         author={Chen, Weihang and Xu, Jing and Xiang, Fanbo and Yuan, Xiaodi and Su, Hao and Chen, Rui},
         journal={IEEE Transactions on Robotics},
         title={General-Purpose Sim2Real Protocol for Learning Contact-Rich Manipulation With Marker-Based Visuotactile Sensors},
         year={2024},
         volume={40},
         number={},
         pages={1509-1526},
         keywords={Sensors;Task analysis;Tactile sensors;Robots;Robot kinematics;Feature extraction;Deformation;Contact-rich manipulation;robot simulation;sim-to-real;tactile sensing},
         doi={10.1109/TRO.2024.3352969}}
@ARTICLE{10027470,
         author={Zhang, Xiaoshuai and Chen, Rui and Li, Ang and Xiang, Fanbo and Qin, Yuzhe and Gu, Jiayuan and Ling, Zhan and Liu, Minghua and Zeng, Peiyu and Han, Songfang and Huang, Zhiao and Mu, Tongzhou and Xu, Jing and Su, Hao},
         journal={IEEE Transactions on Robotics}, 
         title={Close the Optical Sensing Domain Gap by Physics-Grounded Active Stereo Sensor Simulation}, 
         year={2023},
         volume={39},
         number={3},
         pages={2429-2447},
         keywords={Stereo vision;Robots;Robot sensing systems;Optical sensors;Solid modeling;Rendering (computer graphics);Lighting;Active stereovision;depth sensor;robot simulation;sensor simulation;sim-to-real},
         doi={10.1109/TRO.2023.3235591}}
```

