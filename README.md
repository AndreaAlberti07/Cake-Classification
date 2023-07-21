<h1 align="center">Cake Image Classification with MLP and PVMLNet</h1>

<p align="center">
  <b>Developing accurate classification models for cake images</b>
  <br>
  <i>Exploring MLP and PVMLNet approaches</i>
</p>

<p align="center">
  <a href="#results">Results</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#approach">Approach</a> •
  <a href="#features">Features</a> •
  <a href="#conclusion">Conclusion</a> •
  <a href="#usage">Usage</a> •
  <a href="#license">License</a>
</p>

---

## Results

The results demonstrate that the best accuracy of 90% was achieved by utilizing neural features, showcasing the power of leveraging pre-trained CNNs for image classification tasks. In contrast, the utilization of low-level features yielded a comparatively lower accuracy of 31%. These findings highlight the importance of utilizing deeper representations learned by neural networks, which can capture complex patterns and nuances in the cake images.

## Dataset

The dataset used in this project comprises 15 different cake images, each belonging to a specific class. The images were preprocessed and labeled before being used for model training and evaluation.

## Approach

1. **Multi Layer Perceptron (MLP):** A custom MLP model was implemented and trained using the extracted features from the dataset. The model architecture consisted of multiple hidden layers to learn complex patterns in the data.

2. **PVMLNet (Pre-trained CNN):** PVMLNet, a simplified version of AlexNet, was utilized as a pre-trained CNN for transfer learning. The model was fine-tuned using the cake image dataset to adapt it to the specific classification task.

## Features

The following feature types were used in the experiments:

- Low-level features: Color histogram and co-occurrence matrix
- Combination of low-level features

Additionally, neural features were extracted from different layers of the PVMLNet model.

## Conclusion

The project successfully developed classification models for cake images using MLP and a pre-trained CNN (PVMLNet). The results emphasize the effectiveness of neural features extracted from pre-trained CNNs, which significantly outperformed traditional low-level features. By leveraging the power of deep representations learned by CNNs, the models were able to achieve high accuracy in classifying the cake images.

The findings of this project can be valuable for similar image classification tasks where pre-trained CNNs can be employed to boost model performance and accuracy.

## Usage

To use the classification models developed in this project, refer to the provided code and Jupyter notebooks in the repository. 

## License

This project is licensed under the [MIT License](https://github.com/AndreaAlberti07/Cake-Classification/blob/main/LICENSE). Feel free to use and modify the code as per the terms of the license.
