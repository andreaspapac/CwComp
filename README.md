# CwComp
Convolutional Channel-wise Competitive Learning for the Forward-Forward
Algorithm. AAAI 2024

### Abstract
The Forward-Forward (FF) Algorithm has been recently proposed to alleviate the issues of backpropagation (BP) commonly used to train deep neural networks. However, its current formulation exhibits limitations such as the generation of negative data, slower convergence, and inadequate performance on complex tasks. In this paper, we take the main ideas of FF and improve them by leveraging channel-wise competitive learning in the context of convolutional neural networks for image classification tasks. A layer-wise loss function is introduced that promotes competitive learning and eliminates the need for negative data construction. To enhance both the learning of compositional features and feature space partitioning, a channel-wise feature separator and extractor block is proposed that complements the competitive learning process. Our method outperforms recent FF-based models on image classification tasks, achieving testing errors of 0.58%, 7.69%, 21.89%, and 48.77% on MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 respectively. Our approach bridges the performance gap between FF learning and BP methods, indicating the potential of our proposed approach to learn useful representations in a layer-wise modular fashion, enabling more efficient and flexible learning. 

### Files Description
- `predict_main.py`: This script is used for running pre-trained models. It includes functionality for visualizing layer-wise feature maps for each class.
- `train_main.py`: Script for training models based on user-defined configurations.
- `layer_cnn.py`: Contains the implementation of convolutional layer classes and loss function classes.
- `layer_fc.py`: Includes the implementation of fully connected layer classes.
- `datasets.py`: Handles various datasets. (Further description needed).

### Configuration and Usage
The code allows for extensive configuration to cater to different training and evaluation scenarios. Key configurations include:

- `--data_path`: Specify the path to the data.
- `--seed`: Set the random seed for reproducibility.
- `--loss_criterion`: Choose from a variety of loss functions.
- `--dataset`: Select the dataset to use (e.g., MNIST, FMNIST, CIFAR, CIFAR100).
- `--ILT`: Choose the ILT Strategy (e.g., Acc, Fast).
- Additional flags for architecture, predictor settings, class grouping, retraining options, batch size, and more.

### Getting Started
1. Clone the repository.
2. Install required dependencies (pytorch, etc.).
3. Configure your training or prediction parameters in `configure.py`.
4. Run `python train_main.py` to start training or `python predict_main.py` for prediction and visualization.

### Contribution
Contributions to this project are welcome. Please ensure that you follow the code structure and naming conventions for consistency.

### License
This project is licensed under the [MIT License](LICENSE.md).

### Citation
If you use this work in your research, please cite:
To be published in AAAI 2024, Arxiv Preprint:
```bibtex

```

### Contact
For queries, feel free to raise an issue in the GitHub repository or contact the maintainers directly.

---
