# This repository contains the official implementation of the paper "Temporal-coded spiking neural networks with dynamic firing threshold: Learning with event-driven backpropagation".

## 📁 File Structure

```
DTA-TTFS/
├── pretrained/                     # Directory for pre-trained ANN weights
├── models/                         # Model definitions
│   ├── proxy_ann_model.py          # Proxy Model
│   └── snn_directTrain_model.py    # Direct Model
├── utils/                          # Utility functions and data augmentation strategies
│   ├── autoaugment.py         
│   ├── cutout.py          
│   ├── ops.py
├── conventionalANN.py              # ANN model implementation
├── mainFile.py                     # Main training/evaluation script
├── snn_stepwise_inference_model.py  # Stepwise inference model for SNN
```

---

## 🚀 Quick Start

### 1. Prepare Pre-trained ANN Weights
- Download the pre-trained ANN weights and place them in the `./pretrained/` directory. The pre-trained ANN weights are available on the Hugging Face repository:  👉 [Link](https://huggingface.co/wjwei/DTA-TTFS/tree/main).
- You can train your own ANN via **`conventionalANN.py`**.

### 2. Run the Code
```
python mainFile.py
```

### 3. For Testing/Deployment
```
python snn_stepwise_inference_model.py
```
---

## 📄 Citation

If you use this code, please cite the original paper❤:

```bibtex
@inproceedings{wei2023temporal,
  title={Temporal-coded spiking neural networks with dynamic firing threshold: Learning with event-driven backpropagation},
  author={Wei, Wenjie and Zhang, Malu and Qu, Hong and Belatreche, Ammar and Zhang, Jian and Chen, Hong},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={10552--10562},
  year={2023}
}
```
---

## 📧 Contact

For questions or issues regarding the code, please contact the author via 📧wjwei@std.uestc.edu.cn.
