# ASL Alphabet Gestures Recognition 👐🏻🧬

This project implements a machine learning model to recognize and classify American Sign Language (ASL) alphabet gestures using image data.

## 🛠️ Tools & Libraries

This project uses the following key libraries:

* **MediaPipe**: For extracting 3D hand landmarks from images.
* **OpenCV**: For image processing and preprocessing tasks.
* **PyTorch**: To build, train, and evaluate the deep learning model.


## 📂 Dataset

For training the model, a **synthetic ASL alphabet dataset** was used from Kaggle:

🔗 [Synthetic ASL Alphabet Dataset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)

> This dataset contains synthetic images of hand signs representing ASL alphabet letters, including a "Blank" class.

### 🔧 Preprocessing Notes:

* The dataset originally included 26 letters (A–Z) and a **"Blank"** class.
* The **"Z" class was removed** because:
  * The ASL "Z" gesture involves movement.
  * The provided static representation closely resembled the letter **"D"**, leading to confusion.
* The **"Blank" class was renamed to "Nothing"** for better clarity and consistency in training and evaluation.

## 🧠 Model Architecture

The model consists of two main branches:

* **SignImageBranch**: A convolutional multi-scale CNN processing grayscale image inputs (1×224×224).
* **LandmarksBranch**: A dense network that processes hand keypoint landmarks (21×3).

 🧩 Outputs from both branches are fused and passed through a classifier to predict one of **26 classes**:
> * **25 ASL alphabet gestures** (excluding **"Z"**, which involves motion),
> * Plus an additional **"Nothing"** class to represent the absence of a sign.


### 🔧 Architecture Summary

* Total Parameters: **334,362**
* Trainable Parameters: **334,362**
* Total MACs: **44.45 G**
* Input Size: **1×224×224 image + 21×3 landmarks**

## 📈 Model Performance

The model was evaluated on a test set with the following metrics:

* **Accuracy**: 0.80
* **F1 Score (Macro Avg)**: 0.79
* **Precision (Macro Avg)**: 0.80
* **Recall (Macro Avg)**: 0.80

### 📓 Further Evaluation

If you want deeper evaluation, including detailed metrics and the full model architecture, check out the [02_model_predictions.ipynb](./notebooks/02_model_predictions.ipynb) notebook.

