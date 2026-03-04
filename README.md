# rPPG Hybrid Training & Real-time Estimation

Forked from ubicomplab/rPPG-Toolbox.

This repository extends rPPG-Toolbox for hybrid training using the **UBFC-rPPG** and **PURE** datasets and includes **real-time rPPG estimation using an Intel RealSense camera**.

---

## Features

- Hybrid dataset training (**UBFC-rPPG + PURE**)
- Custom dataset loader
- TSCAN-based rPPG model training
- Real-time rPPG inference
- Intel RealSense camera support

---


## Dataset

Datasets are **not included** in this repository.

Expected dataset structure:
/home/jung/rPPG_MIX
├── UBFC
└── PURE

---


## Real-time rPPG (Intel RealSense)

Real-time heart rate estimation using a RealSense camera:

Requirements:

- Intel RealSense camera
- `pyrealsense2`

---


## Acknowledgment

This project is based on the original **rPPG-Toolbox** by ubicomplab.
