# ManiSkill-ViTac 2025

### Vision and Tactile Sensing Challenge for Manipulation Skill Learning

[![Challenge Website](https://img.shields.io/badge/View-Challenge_Website-blue)](https://ai-workshops.github.io/maniskill-vitac-challenge-2025/)
[![Discord](https://img.shields.io/badge/Join-Discord-7289DA)](https://discord.gg/CKucPQxQPr)

## 🚀 Latest Updates
- **2025/01/21**: Update the pretain weight in huggingface [ManiSkill-ViTac2025_ckpt](https://huggingface.co/cyliizyz/ManiSkill-ViTac2025_ckpt).
- **2025/01/02**: Refactor the entire repository, added render_rgb options, and visualization during the evaluation. For details, please refer to the repository. Stage 1 starts! 🚀
- **2024/12/08**: Refactor code structure for better readability.
- **2024/11/16**: Track 2 is online! 
- **2024/10/22**: Track 3 is now available! Featuring sensor structure optimization.

## 📋 Overview

ManiSkill-ViTac 2025 is a challenge focused on developing advanced manipulation skills using vision and tactile sensing. The challenge consists of three tracks:

- **Track 1: Visuotactile Manipulation**
  - Manipulation tasks with tactile sensing
  - Input Information: Tactile information only

- **Track 2: Tactile-Vision-Fusion Manipulation**
  - Manipulation with enhanced tactile feedback
  - Input Information: Tactile information + depth with semantic segmentation

- **Track 3: Sensor Structure Design**
  - Sensor structure optimization
  - Design Content: Design the shape of the silicone for the GelSight Mini and the distribution of the markers

## 🛠️ Setup Guide

### Prerequisites

**System Requirements**
- Python 3.8.x - 3.11.x
- GCC 7.2 or higher (Linux)
- CUDA Toolkit 11.8 or higher
- [Git LFS](https://git-lfs.github.com/)

**Core Dependencies**
- [SAPIEN v3.0.0b1](https://github.com/haosulab/SAPIEN/releases/tag/3.0.0b1)
- [SapienIPC](https://github.com/Rabbit-Hu/sapienipc-exp)

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/cyliizyz/ManiSkill-ViTac2025.git
   cd ManiSkill-ViTac2025
   ```

2. **Setup Conda Environment**
   ```bash
   conda env create -f environment.yaml
   conda activate mani_vitac
   ```

3. **Install SapienIPC**
   - Follow instructions in the [SapienIPC README](https://github.com/Rabbit-Hu/sapienipc-exp/blob/main/README.md)

⚠️ **Important Note**:
- Install SAPIEN v3.0.0b1 before SapienIPC for compatibility

## 🎯 Challenge Tracks

### Track 1: Visuotactile Manipulation

**Input Information**
- Tactile information only

**Setup**
- Modify network structure in `Track_1/solutions/`
- Save model in `Track_1/solutions/policies.py`

**Training**
```bash
# Peg insertion training
python Track_1/scripts/universal_training_script.py --cfg Track_1/configs/parameters/peg_insertion.yaml

# Open lock training
python Track_1/scripts/universal_training_script.py --cfg Track_1/configs/parameters/long_open_lock.yaml
```

**Evaluation & Submission**
```bash
# Evaluate peg insertion
python Track_1/scripts/peg_insertion_sim_evaluation.py \
    --team_name [your_teamname] \
    --model_name [your_model_name] \
    --policy_file_path [your_best_model_path]

# Evaluate open lock
python Track_1/scripts/open_lock_sim_evaluation.py \
    --team_name [your_teamname] \
    --model_name [your_model_name] \
    --policy_file_path [your_best_model_path]
```

### Track 2: Tactile-Vision-Fusion Manipulation

**Input Information**
- Tactile information + depth with semantic segmentation

**Setup**
- Modify network structure in `Track_2/solutions/`
- Save model in `Track_2/solutions/policies.py`

**Training**
```bash
python Track_2/scripts/universal_training_script.py \
    --cfg Track_2/configs/parameters/peg_insertion_v2_points.yaml
```

**Evaluation & Submission**
```bash
python Track_2/scripts/peg_insertion_v2_sim_evaluation.py \
    --team_name [your_teamname] \
    --model_name [your_model_name] \
    --policy_file_path [your_best_model_path]
```

### Track 3: Sensor Structure Design

**Design Content**
- Design the shape of the silicone for the GelSight Mini
- Design the distribution of the markers

**Prerequisites**
- Install PyMesh (required for mesh processing):
   ```bash
   # Ensure mani_vitac environment is activated
   conda activate mani_vitac

   # System dependencies (Ubuntu/Debian)
   sudo apt-get install \
       libeigen3-dev \
       libgmp-dev \
       libgmpxx4ldbl \
       libmpfr-dev \
       libboost-dev \
       libboost-thread-dev \
       libtbb-dev

   # Build and Install PyMesh into mani_vitac environment
   git clone https://github.com/PyMesh/PyMesh.git
   cd PyMesh
   python setup.py build
   python setup.py install
   cd ..
   ```

**Setup**
1. Design silicone component using modeling software
2. Process the model:
   ```bash
   # Adjust orientation
   python Track_3/tools/translate_STL.py
   
   # Generate environment-compatible model
   python Track_3/tools/generate_mesh.py
   ```
3. Update `tac_sensor_meta_file` in `Track_3/configs/parameters/peg_insertion.yaml`
4. Configure markers in `Track_3.envs.tactile_sensor_sapienipc.VisionTactileSensorSapienIPC`

**Training & Evaluation**
```bash
# Training
python Track_3/scripts/universal_training_script.py \
    --cfg Track_3/configs/parameters/peg_insertion.yaml

# Evaluation
python Track_3/scripts/peg_insertion_sim_evaluation.py \
    --team_name [your_teamname] \
    --model_name [your_model_name] \
    --policy_file_path [your_best_model_path]
```

## 📊 Competition Resources

- **Leaderboard**: [View Rankings](https://ai-workshops.github.io/maniskill-vitac-challenge-2025/#leaderboard)
- **Real Robot Demo**: Available in `real_env_demo/`
- **GelSightMini Sensor**: [Code Repository](https://github.com/RVSATHU/gelsight_mini_ros)

## 📞 Contact & Support

- **Discord**: [Join Community](https://discord.gg/CKucPQxQPr)
- **Email**: [maniskill.vitac@gmail.com](mailto:maniskill.vitac@gmail.com)

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