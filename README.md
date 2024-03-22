# CwComp - Supplementary Material Section
---

## Convolutional Channel-wise Competitive Learning for the Forward-Forward Algorithm - AAAI 2024

### Abstract
The Forward-Forward (FF) Algorithm has been recently proposed to alleviate the issues of backpropagation (BP) commonly used to train deep neural networks. However, its current formulation exhibits limitations such as the generation of negative data, slower convergence, and inadequate performance on complex tasks. In this paper, we take the main ideas of FF and improve them by leveraging channel-wise competitive learning in the context of convolutional neural networks for image classification tasks. A layer-wise loss function is introduced that promotes competitive learning and eliminates the need for negative data construction. To enhance both the learning of compositional features and feature space partitioning, a channel-wise feature separator and extractor block is proposed that complements the competitive learning process. Our method outperforms recent FF-based models on image classification tasks, achieving testing errors of 0.58%, 7.69%, 21.89%, and 48.77% on MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 respectively. Our approach bridges the performance gap between FF learning and BP methods, indicating the potential of our proposed approach to learn useful representations in a layer-wise modular fashion, enabling more efficient and flexible learning. 

### Supplementary Material

#### Source Code Availability
The complete source code used in this research, including scripts for training and prediction, is available in this repository to ensures the reproducibility of our results, and provide a practical foundation for further research and development based on our work.

#### Appendix
The following sections provide a brief overview of the supplementary material included in this project. For a more in-depth discussion and additional details, please refer to our Arxiv version.

##### Visual Analysis of CFSE-Driven Feature Space Separation
We provide additional qualitative results showcasing feature map visualizations produced by the CFSE layers of our model on MNIST dataset test inputs. The visualizations demonstrate how our CFSE architecture, combined with the CwC loss, effectively extracts and separates features along the channel dimension, with higher activations for the correct class. For comprehensive analysis and further insights, see the detailed discussion in our Arxiv paper.

- Feature maps for Target-class: 3 ![three](https://github.com/andreaspapac/CwComp/assets/154099956/937dbb19-f04d-4e52-99d0-651aa63d80cb)

- Feature maps for Target-class: 6 ![six](https://github.com/andreaspapac/CwComp/assets/154099956/2c107cdf-0829-47cf-af6c-1999dc3e2814)

- Feature maps for Target-class: 9 ![nine](https://github.com/andreaspapac/CwComp/assets/154099956/c314111f-ead4-4f26-a406-f1c99097027b)

##### Layer-wise Training Convergence Rates
We compare the training convergence rates of our CFSE_CwC models with the reproduced FF algorithm. Our models show significantly faster convergence and lower testing errors, demonstrating the efficacy of our approach. The complete analysis and comparative discussion can be found in the extended version on Arxiv.

- Convergence Plots of the FF-rep* Model ![FFrep_th075](https://github.com/andreaspapac/CwComp/assets/154099956/17f01da9-fdd1-406f-90d5-42ee3017f76a)

- Convergence Plots of CFSE_CwC Model ![CFSE_CwC_th075](https://github.com/andreaspapac/CwComp/assets/154099956/18c40c1b-efd8-477e-805c-559d7dded852)

The impact of the ILT strategy on testing errors of different predictors and the importance of each CFSE layer in the network's overall performance is also discussed.

##### Model Configurations
For reproducibility and clarity, we detail the configurations of models using CFSE, FF-CNN, and FF-FC architectures, including the Fully Connected (FC) layer configuration.

- FC layer configuration for CNN models
  ![image](https://github.com/andreaspapac/CwComp/assets/154099956/eacb4746-c661-4914-9112-6ba3023a6711)
- Configurations of CFSE and FF-CNN models 
![image](https://github.com/andreaspapac/CwComp/assets/154099956/8fe1ef27-3594-4e27-9219-7def3508395e)
- Configurations of the FF-rep* model 
![image](https://github.com/andreaspapac/CwComp/assets/154099956/ca5de74e-69a3-4449-af2b-5690b5bdfba0)

---
## Source Code

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

Presented in AAAI 2024, Arxiv Preprint:

```
@misc{papachristodoulou2023convolutional,
      title={Convolutional Channel-wise Competitive Learning for the Forward-Forward Algorithm}, 
      author={Andreas Papachristodoulou and Christos Kyrkou and Stelios Timotheou and Theocharis Theocharides},
      year={2023},
      eprint={2312.12668},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

ESANN 2023 Proceedings:

```
@inproceedings{papachristodoulou_2023_10781728,
  author = {Papachristodoulou, Andreas and Kyrkou, Christos and Timotheou, Stelios and Theocharides, Theocharis},
  title = {{Introducing Convolutional Channel-wise Goodness in Forward-Forward Learning}},
  booktitle = {Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)},
  year = {2023},
  publisher = {i6doc.com publ.},
  address = {Bruges, Belgium},
  doi = {10.14428/esann/2023.ES2023-121},
  url = {https://doi.org/10.14428/esann/2023.ES2023-121},
  isbn = {978-2-87587-088-9},
  note = {ESANN 2023 proceedings, Bruges (Belgium) and online event, 4-6 October 2023. Available from http://www.i6doc.com/en/.}
}

```

### Contact
For queries, feel free to raise an issue in the GitHub repository or contact the maintainers directly.

---
