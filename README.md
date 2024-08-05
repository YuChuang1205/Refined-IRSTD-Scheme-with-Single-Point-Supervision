## The code for paper "Refined Infrared Small Target Detection Scheme with Single-Point Supervision"  

### **The proposed scheme won the third place in the "ICPR 2024 Resource-Limited Infrared Small Target Detection Challenge Track 1: Weakly Supervised Infrared Small Target Detection"**  

### This code is based on Label Evolution with Single Point Supervision (LESPS) ([[link](https://github.com/XinyiYing/LESPS?tab=readme-ov-file)]) by Dr. Xinyi Ying. Thanks to the previous work.

### In this code, we have integrated multiple networks (ACM, ALCNet, MLCL-Net, ALCL-Net, DNA-Net, GGL-Net) for use. For MLCL-Net, there are two versions in the code: MLCL-Net (small) and MLCL-Net (base). For details, see the description in the corresponding model code file. It is worth noting that we use MLCL-Net (small) in this paper.

### For the data set creation, you can use "centroid_anno.m" and "coarse_anno.m". For details, please refer to the details in LESPS ([[link](https://github.com/XinyiYing/LESPS?tab=readme-ov-file)])

### It is worth mentioning that GGL-Net ([[paper](https://ieeexplore.ieee.org/abstract/document/10230271)]) has extremely excellent performance in this task. For specific performance comparisons of each network, please refer to the paper. 

### If you have any questions, please feel free to ask. I will reply in time on github.
