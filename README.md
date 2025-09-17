# Temporal-coded spiking neural networks with dynamic firing threshold: Learning with event-driven backpropagation

This repository provides the official implementation of the paper **"Temporal-coded spiking neural networks with dynamic firing threshold: Learning with event-driven backpropagation"** (ICCV 2023).

## 📁 File Structure

```
DTA-TTFS/
├── pretrained/                     # Pre-trained ANN model weights
├── models/                         # Model architectures
│   ├── proxy_ann_model.py          # Proxy ANN model definition
│   └── snn_directTrain_model.py    # Direct training SNN model
├── utils/                          # Utilities and data augmentation
│   ├── autoaugment.py              
│   ├── cutout.py                   
│   └── ops.py                    
├── conventionalANN.py              # Conventional ANN training script
├── mainFile.py                     # Main training and evaluation script
└── snn_stepwise_inference_model.py # Stepwise SNN inference for deployment
```

---

## 🚀 Quick Start

### 1. Prepare Pre-trained Weights
- **Option A**: Download pre-trained ANN weights from our [Hugging Face repository](https://huggingface.co/wjwei/DTA-TTFS/tree/main) and place them in the `./pretrained/` directory.
- **Option B**: Train your own ANN model using **`conventionalANN.py`**.


### 2. Run Main Training
```
python mainFile.py
```

### 3. For Deployment & Testing
```
python snn_stepwise_inference_model.py
```

---

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wei2023temporal,
  title={Temporal-coded spiking neural networks with dynamic firing threshold: Learning with event-driven backpropagation},
  author={Wei, Wenjie and Zhang, Malu and Qu, Hong and Belatreche, Ammar and Zhang, Jian and Chen, Hong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10552--10562},
  year={2023}
}
```

---

## 📧 Contact

For questions regarding this implementation, please contact: **Wenjie Wei** 📧 wjwei@std.uestc.edu.cn
