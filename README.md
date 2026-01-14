# Prior knowledge-embedded first-layer interpretable paradigm for rail transit vehicle human-computer collaboration fault monitoring

## 🔍 Overview

Rail transit vehicles endure large loads, high speeds, and harsh environment, leading to component failure. The first-layer interpretable paradigm (FLIP) embeds human prior knowledge into smart equipment, which is one of intelligent paradigms guided by customized manufacturing and embodied intelligence. It consists of first-layer interpretable modules, backbones, loss metrics. However, existing efforts rely on single-source information, an absence of interpretable backbones, an inability to feature fusion, thereby struggling with multi-excitation, coupled signals. To bridge this gap, a FLIP-based one-stage multi-view capsule fusion network (PIFCapsule) is proposed. Firstly, a signal processing prior-empowered first-layer interpretable module is devised to realize automatic parameter optimization and highlight the complementarity between multi-view features from different signal processing algorithms. Secondly, an interpretable capsule network serves as the backbone. To overcome the inefficiency and shortage of information fusion, an efficient attention fusion routing (AFR) is proposed to reduce the parameters (about 5.72 times) and the complexity (about 2.93 times) in contrast to the vanilla capsule-based network. In response to the lack of physics-based constraints during training, a noise threshold amplitude ratio (NTAR) is posed as a regularization, which enhances weak periodic transient pulses by suppressing learned noises. The effectiveness and reliability are verified through three real-world rail transit vehicle datasets: PIFCapsule outperforms the state-of-the-art by 6.77% in accuracy with only ten samples. Given the lightweight nature, it holds substantial promise to be deployed in intelligent edge devices. Code is available at https://github.com/liguge/PIFCapsule.

![image](https://github.com/liguge/PIFCapsule/blob/main/image/Fig1.png)

### Key Features

- **physics-informed Multi-view Feature Fusion**: Integrates wavelet transform, STFT, and blind convolution to extract complementary time/frequency domain features.

- **Efficient Attention Fusion Routing (AFR)**: Reduces parameters by 82% and computational complexity by 66% compared to vanilla capsule networks.

- **Physics-Informed Regularization (NTAR)**: Automatically suppresses noise and enhances weak fault features without prior fault period information.

- **Strong Interpretability**: First-layer weight initialization and routing coefficient visualization enable transparent fault feature tracing.

- **Small-Sample Superiority**: Achieves 93.56% average accuracy with only 10 samples per class, outperforming SOTA models by 6.77%.

## 📚 Paper

**Title**: Prior knowledge-embedded first-layer interpretable paradigm for rail transit vehicle human-computer collaboration fault monitoring  

**Authors**: **Chao He**, Hongmei Shi*, Jing-Xiao Liao, Bin Liu, Qiuhai Liu, Jianbo Li, Zujun Yu  

**Journal**: Journal of Industrial Information Integration

**Paper Link:**  https://doi.org/10.1016/j.jii.2026.101068

## 🛠️ Installation

### Prerequisites

- Python 3.8+

- PyTorch 2.5+

- CUDA 11.7+ (for GPU acceleration)

- Other dependencies:

## 📊 Datasets

We validate PIFCapsule on three real-world rail transit vehicle datasets. Due to data confidentiality, we provide **data formats and simulation scripts** for reproduction.

### Dataset Details

|Dataset|Source|Speed Range|Sampling Frequency|Health States|
|---|---|---|---|---|
|BJTU₁|High-speed train traction motor|200-350 km/h|100 kHz|Normal (N), Inner Fault (IF), Outer Fault (OF), Ball Fault (BF)|
|BJTU₂|Heavy-haul freight train wheelset|60-180 km/h|16 kHz|7 faults (IRP/ORP/REP/REC/CF/CHC/CFs) + Healthy (H)|
|BJTU₃|Subway bogie gearbox|20-60 Hz + 0/10 kN|64 kHz|9 states (Normal/GCT/GWT/GMT/GCPT/BIR/BOR/BC/BFE)|
### Data Preparation

##### For real data, organize into the following structure:

```Plain Text

Datapaper/
├── G/
│   ├── 200.npy/
│   ├── 
│   ├──   
│   ├──    
│   ├── 
│   └── 
├── /
└── /
```

## 📁 Code Structure

```Plain Text
PIFCapsule/
├── CNN_Datasets/               # CNN dataset related
│   ├── R_NA/                  # Specific dataset
│   │   ├── datasets/
│   │   │   ├── G.py          # Data generation or processing
│   │   │   └── __init__.py
│   │   └── __init__.py
│   └── __init__.py
├── DataPaper/                 # Data paper related
│   ├── G/
│   │   └── download.md       # Data download instructions
│   └── __init__.py
├── datasets/                  # Core dataset module
│   ├── MatrixDatasets.py     # Matrix dataset processing
│   ├── SequenceDatasets.py   # Sequence dataset processing
│   ├── con_dataset.py        # Generic dataset processing
│   ├── matrix_aug.py         # Matrix data augmentation
│   ├── process_data_2.py     # Data preprocessing
│   ├── sequence_aug.py       # Sequence data augmentation
│   └── __init__.py
├── models/                    # Core model module
│   ├── CapsNetfusion.py      # Main capsule network fusion model
│   ├── layers.py             # Capsule network layer definitions
│   ├── blind.py              # Blind convolution processing
│   ├── weight_init.py        # Weight initialization
│   └── __init__.py
├── utils/                     # Utility functions
│   ├── logger.py             # Logging
│   ├── loss.py               # Loss functions
│   ├── train_utils.py        # Training main code
│   └── __init__.py
├── checkpoint/                # Model checkpoints
├── train.py                   # Main training script
├── README.md                  # Project documentation
└── __init__.py
```

## 🎯 Main Results

### Performance on Small-Sample Scenarios (10 samples per class)

|Dataset|PIFCapsule|EfficientCapsule|Vanilla Capsule|
|---|---|---|---|
|BJTU₁|98.02%|83.52%|80.28%|
|BJTU₂|90.82%|42.02%|43.46%|
|BJTU₃|98..31%|79.58%|90.78%|
### Parameter & Complexity Comparison

|Model|Parameters (MB)|FLOPs (MB)|Accuracy|
|---|---|---|---|
|PIFCapsule|0.48|22.73|95.72%|
|Vanilla Capsule|2.76|68.80|68.37%|
|EfficientCapsule|2.76|68.80|71.51%|
## 🌟 Key Innovations

1. **First-layer interpretable paradigm**: Embeds signal processing prior knowledge into the first layer, guiding model optimization.

2. **AFR Mechanism**: Enables efficient cross-modal fusion without additional bottleneck layers.

3. **NTAR Regularization**: Adaptive noise suppression for high-noise rail transit scenarios.

4. **Lightweight Design**: Suitable for edge deployment in rail transit monitoring systems.

## 📝 Citation

If you use PIFCapsule in your research, please cite our paper:

```Plain Text
Liu, Jianbo Li and ZuJun Yu. Prior knowledge-embedded first-layer interpretable paradigm for rail transit vehicle human-computer collaboration fault monitoring[J]. Journal of Industrial Information Integration, 2026: 101068. doi: 10.1016/j.jii.2026.101068.


@article{he2025pifcapsule,
  title={Human prior knowledge-embedded first-layer interpretable paradigm for rail transit vehicle human-computer collaboration monitoring},
  author={He, Chao and Shi, Hongmei and Liao, Jing-Xiao and Liu, Qiuhai and Li, Jianbo and Yu, Zujun},
  journal={Journal of Industrial Information Integration},
  volume={XX},
  number={XX},
  pages={100XXX},
  year={2025},
  doi={10.1016/j.jii.2026.101068},
  publisher={Elsevier}
}
```

## 🎯 Future Work

- Support transfer learning across different rail transit vehicles.

- Extend to more signal processing modules (e.g., wavelet packet transform).

- Optimize for edge computing with TensorRT acceleration.

## 📧 Contact

For questions or issues, please contact:

- Chao He: chaohe#bjtu.edu.cn

## ❤️Thanks

The authors would like to express their sincere gratitude to Jing-Xiao Liao, Bin Liu, Meng Wang, Tianfu Li, Zhibin Zhao, and the anonymous editors and reviewers for their valuable help and support.

---

*This repository is maintained by the Institute of Intelligent Inspection Technology for Rail Transit at Beijing Jiaotong University. We welcome contributions and feedback!*