<div align="center">
<h1>BRIDGE - Building Reinforcement-Learning Depth-to-Image Data Generation Engine for Monocular Depth Estimation</h1>

[**Dingning Liu**](https://github.com/lnbxldn)<sup>1,2</sup> · [**Haoyu Guo**](https://github.com/ghy0324)<sup>1</sup> · [**Jingyi Zhou**](https://github.com/zjyaccount)<sup>1</sup> · [**Tong He**](https://tonghe90.github.io/)<sup>1&dagger;</sup>

<sup>1</sup>Shanghai AI Lab &emsp; <sup>2</sup>Fudan University

&dagger;Corresponding author

<a href="https://arxiv.org/abs/2509.25077"><img src='https://img.shields.io/badge/arXiv-BRIDGE-red' alt='Paper PDF'></a>

<a href='https://dingning-liu.github.io/bridge.github.io/'><img src='https://img.shields.io/badge/Project_Page-BRIDGE-green' alt='Project Page'></a>

<a href='https://huggingface.co/spaces/Dingning/Bridge'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>

</div>

---
Official implementation of  of BRIDGE: Building Reinforcement-Learning Depth-to-Image Data Generation Engine for Monocular Depth Estimation
![teaser](assets/teaser.png)

## 📰 News

- **[2025-09-30] 🚀🚀🚀 We published BRIDGE on [arXiv](https://arxiv.org/abs/2509.25077) and demos on huggingface! Try our [DEMO](https://huggingface.co/spaces/Dingning/Bridge)!!!**
- [2025-09-30] **🎉🎉🎉** We released the model [checkpoint](https://huggingface.co/Dingning/BRIDGE) on huggingface.

## 🛫Overview

1. We present BRIDGE, an RL-optimized, large-scale Depth-to-Image (D2I) data engine. It generates massive, high-quality RGB-D data to address critical Monocular Depth Estimation (MDE) training challenges and foster robust real-world performance.

   Our main contributions are summarized as follows:

   1. **An efficient RL-driven D2I data engine:** BRIDGE efficiently generates over 20 million diverse, high-quality RGB-D data from synthetic depth, alleviating data scarcity and quality issues.
   2. **A novel hybrid depth supervision strategy:** We introduce a hybrid training strategy combining generated RGB with high-precision ground truth and teacher pseudo-labels, enhancing geometric knowledge learning.
   3. **Superior performance and high training efficiency:** BRIDGE achieves SOTA MDE performance across benchmarks with significantly less data (20M vs. 62M), demonstrating excellent detail capture and robustness.


## 📀Pre-trained Models
Download the checkpoint from [huggingface](https://huggingface.co/Dingning/BRIDGE/resolve/main/bridge.pth) and put it under the `checkpoints` directory.

## 🏋️Prepraration

```bash
git clone https://github.com/lnbxldn/Bridge.git
cd Bridge
pip install -r requirements.txt
```

## 💻Inference 

```python
import cv2
import torch
import numpy as np
from bridge.dpt import Bridge 
model = Bridge()
model.load_state_dict(torch.load(f'checkpoints/bridge.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img)  
```

## 🔍Citation

If you find this project useful, please citing:

```bibtex
@misc{Liu2025BRIDGE,
  title={BRIDGE - Building Reinforcement-Learning Depth-to-Image Data Generation Engine for Monocular Depth Estimation},
  author={Liu, Dingning and Guo, Haoyu and Zhou, Jingyi and He, Tong},
  year={2025},
  eprint={2509.25077},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2509.25077},
}
```

