# ManiSkill-ViTac 2025

### Vision and Tactile Sensing Challenge for Manipulation Skill Learning

[![Challenge Website](https://img.shields.io/badge/View-Challenge_Website-blue)](https://ai-workshops.github.io/maniskill-vitac-challenge-2025/)
[![Discord](https://img.shields.io/badge/Join-Discord-7289DA)](https://discord.gg/CKucPQxQPr)

## 🚀 Latest Updates

- **2024/11/16**: Track 2 is online! 
- **2024/10/22**: Track 3 is now available! Featuring sensor structure optimization.

## 📋 Overview

ManiSkill-ViTac 2025 is a challenge focused on developing advanced manipulation skills using vision and tactile sensing. The challenge consists of three tracks, each targeting different aspects of robotic manipulation.

## 🛠️ Prerequisites

### System Requirements
- Python 3.8.x - 3.11.x
- GCC 7.2 or higher (Linux)
- CUDA Toolkit 11.8 or higher
- [Git LFS](https://git-lfs.github.com/)

### Dependencies
- [SAPIEN v3.0.0b1](https://github.com/haosulab/SAPIEN/releases/tag/3.0.0b1)
- [SapienIPC](https://github.com/Rabbit-Hu/sapienipc-exp)

## 📥 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cyliizyz/ManiSkill-ViTac2025.git
   cd ManiSkill-ViTac2025
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate mani_vitac
   ```

3. Install SapienIPC following the instructions in its [README](https://github.com/Rabbit-Hu/sapienipc-exp/blob/main/README.md).

⚠️ *Important*: SAPIEN v3.0.0b1 must be installed before installing SapienIPC to ensure compatibility. Using a different version of SAPIEN may cause unexpected issues.

## 🎯 Challenge Tracks

### Track 1
- Modify the network structure in [solutions](Track_1/solutions) and save the model in [policies.py](Track_1/solutions/policies.py)
- **Important:** Files [peg_insertion_sim_evaluation.py](Track_1/scripts/peg_insertion_sim_evaluation.py) and [open_lock_sim_evaluation.py](Track_1/scripts/open_lock_sim_evaluation.py) **must remain unmodified**. MD5 hash checks will verify compliance.

#### Training Example
```bash
# Example policy for peg insertion
python Track_1/scripts/universal_training_script.py --cfg Track_1/configs/parameters/peg_insertion.yaml
# Example policy for open lock
python Track_1/scripts/universal_training_script.py --cfg Track_1/configs/parameters/long_open_lock.yaml
```

#### Submission
```bash
python Track_1/scripts/peg_insertion_sim_evaluation.py --team_name [your_teamname] --model_name [your_model_name] --policy_file_path [your_best_model_path]
python Track_1/scripts/open_lock_sim_evaluation.py --team_name [your_teamname] --model_name [your_model_name] --policy_file_path [your_best_model_path]
```

Submit evaluation logs to [maniskill.vitac@gmail.com](mailto:maniskill.vitac@gmail.com)

### Track 2
- Modify the network structure in [solutions](Track_2/solutions) and save the model in [policies.py](Track_2/solutions/policies.py)
- **Important:** Files [peg_insertion_v2_sim_evaluation.py](Track_2/scripts/peg_insertion_v2_sim_evaluation.py)  **must remain unmodified**. MD5 hash checks will verify compliance.

#### Training Example
```bash
# Example policy for peg insertion v2
python Track_2/scripts/universal_training_script.py --cfg Track_2/configs/parameters/peg_insertion_v2_points.yaml
```

#### Submission
```bash
python Track_2/scripts/peg_insertion_v2_sim_evaluation.py --team_name [your_teamname] --model_name [your_model_name] --policy_file_path [your_best_model_path]
```

Submit evaluation logs to [maniskill.vitac@gmail.com](mailto:maniskill.vitac@gmail.com)

### Track 3
- Modify the network structure in [solutions](Track_3/solutions) and save the model in [policies.py](Track_3/solutions/policies.py)
- Keep [peg_insertion_sim_evaluation.py](Track_3/scripts/peg_insertion_sim_evaluation.py) unchanged (MD5 hash checks apply)

Track 3 focuses on optimizing the sensor structure. Use modeling software to design the silicone component and import it for evaluation. Refer to the documentation for more details.

After modeling:
1. Use [translate_STL.py](Track_3/tools/translate_STL.py) to adjust orientation
2. Run [generate_mesh.py](Track_3/tools/generate_mesh.py) to create an environment-compatible model, saving it in [assets](Track_3/assets)
3. Update `tac_sensor_meta_file` in [peg_insertion.yaml](Track_3/configs/parameters/peg_insertion.yaml) with the generated folder name
4. Modify parameters within `Track_3.envs.tactile_sensor_sapienipc.VisionTactileSensorSapienIPC` to create markers

#### Training Example
```bash
# Example policy for peg insertion
python Track_3/scripts/universal_training_script.py --cfg Track_3/configs/parameters/peg_insertion.yaml
```

#### Submission
```bash
python Track_3/scripts/peg_insertion_sim_evaluation.py --team_name [your_teamname] --model_name [your_model_name] --policy_file_path [your_best_model_path]
```

Submit evaluation logs and design documentation to [maniskill.vitac@gmail.com](mailto:maniskill.vitac@gmail.com)


## 📊 Leaderboard
Track your progress on the [official leaderboard](https://ai-workshops.github.io/maniskill-vitac-challenge-2025/#leaderboard).

## 🤖 Real Robot Implementation
- Implementation code available in `real_env_demo/`
- GelSightMini sensor code: [gelsight_mini_ros](https://github.com/RVSATHU/gelsight_mini_ros)

## 📞 Contact
- Discord: [Join our community](https://discord.gg/CKucPQxQPr)
- Email: [maniskill.vitac@gmail.com](mailto:maniskill.vitac@gmail.com)

## 📚 Citations

```bibtex
@ARTICLE{chen2024tactilesim2real,
    author={Chen, Weihang and Xu, Jing and Xiang, Fanbo and Yuan, Xiaodi and Su, Hao and Chen, Rui},
    journal={IEEE Transactions on Robotics},
    title={General-Purpose Sim2Real Protocol for Learning Contact-Rich Manipulation With Marker-Based Visuotactile Sensors},
    year={2024},
    volume={40},
    pages={1509-1526},
    doi={10.1109/TRO.2024.3352969}
}

@ARTICLE{10027470,
    author={Zhang, Xiaoshuai and Chen, Rui and Li, Ang and Xiang, Fanbo and Qin, Yuzhe and Gu, Jiayuan and Ling, Zhan and Liu, Minghua and Zeng, Peiyu and Han, Songfang and Huang, Zhiao and Mu, Tongzhou and Xu, Jing and Su, Hao},
    journal={IEEE Transactions on Robotics},
    title={Close the Optical Sensing Domain Gap by Physics-Grounded Active Stereo Sensor Simulation},
    year={2023},
    volume={39},
    number={3},
    pages={2429-2447},
    doi={10.1109/TRO.2023.3235591}
}
```