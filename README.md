Offroad Semantic Segmentation

This project implements a real-time semantic segmentation model for off-road autonomous navigation using SegFormer-B1. Optimized for the Duality AI challenge, it achieves high accuracy on desert terrain while maintaining an inference speed of <50 ms per image.

Project Overview
•	Model Architecture: SegFormer-B1 (Pre-trained on ADE20K).
•	Objective: Segment off-road scenes into navigational classes (Obstacles, Vegetation, Landscape, etc.).
•	Constraint: Inference speed must be < 50ms per image.
•	Input Resolution: 512x512 (Training) / 640x640.

Dataset & Class Mapping
The dataset consists of synthetic off-road images. Although the problem statement listed 10 classes, the dataset contained only 6 unique classes. To stabilize training, we mapped the sparse IDs to a continuous range 0-5.

| Original ID | Class Name        | New Mapped ID |
| :---        | :---              | :---          |
| 0           | Background        | 0             |
| 1           | Sky               | 1             |
| 2           | Ground            | 2             |
| 3           | Vegetation        | 3             |
| 27          | Obstacles (Rocks) | 4             |
| 39          | Landscape         | 5             |

Note: The "Ground" class (ID 2) was excluded from final IoU calculations because it was absent in the test set.

Installation
Ensure you have Python 3.8+ and a GPU-enabled environment.
1.	Clone the repository
2.	For training in an Anaconda Enviroment, run the following code for training and testing the model.
Bash
pip install torch torchvision transformers evaluate opencv-python albumentations tqdm scipy


