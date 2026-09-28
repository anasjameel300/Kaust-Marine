# Dual-Benchmark Scientific Leaderboard: Controlled vs. Real-World Red Sea

| Architecture | Paradigm | Controlled (Fish-Vista) | Real-World (Wild Red Sea) | Robustness Drop (Δ) | GPU Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **YOLO11s-cls** | Edge Real-Time CNN | **95.26%** | **14.00%** | -81.26% | 160.1 ms |
| **ResNet50d** | Classic Residual CNN | **97.81%** | **17.20%** | -80.61% | 6.7 ms |
| **MobileNetV4-Small** | Edge Optimized CNN | **81.10%** | **7.60%** | -73.50% | 2.9 ms |
| **ConvNeXt-Tiny** | Large-Kernel ConvNet | **80.27%** | **12.00%** | -68.27% | 5.0 ms |
| **Swin-Tiny** | Hierarchical ViT | **97.53%** | **23.20%** | -74.33% | 6.2 ms |
| **DINOv2-Small + LoRA** | Vision Foundation Model (PEFT) | **98.08%** | **23.20%** | -74.88% | 4.7 ms |
| **BioCLIP (ViT-B) + LoRA** | Biological VFM (PEFT) | **96.44%** | **51.20%** | -45.24% | 7.9 ms |

*Generated automatically from 7 evaluated models.*
