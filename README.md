# BRIDGE - Building Reinforcement-Learning Depth-to-Image Data Generation Engine for Monocular Depth Estimation
[**Dingning Liu**](https://github.com/lnbxldn)<sup>1,2</sup> · [**Haoyu Guo**](https://github.com/ghy0324)<sup>1</sup> · [**Jingyi Zhou**](https://github.com/zjyaccount)<sup>1</sup> · [**Tong He**](https://tonghe90.github.io/)<sup>1&dagger;</sup>

<sup>1</sup>Shanghai AI Lab &emsp; <sup>2</sup>Fudan University
&dagger;Corresponding author
<br>
Official implementation of  of BRIDGE: Building Reinforcement-Learning Depth-to-Image Data Generation Engine for Monocular Depth Estimation
![teaser](assets/teaser.png)


## Pre-trained Models

## Inference 

```python
import cv2
import torch
import numpy as np
from bridge.dpt import Bridge 
model = Bridge()
model.load_state_dict(torch.load(f'bridge.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img)  
```

