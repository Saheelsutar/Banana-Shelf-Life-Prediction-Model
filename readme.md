# Banana Shelf Life Prediction Using Deep Learning

## Overview

This project presents banana shelf-life prediction system using Deep Learning and Transfer Learning. The system classifies bananas into four ripeness stages:

* Unripe
* Ripe
* Overripe
* Rotten

Based on the predicted ripeness stage, the model estimates the remaining shelf life of the banana.

The model is built using EfficientNetB0 with transfer learning and achieves high classification performance on unseen test data.

---

## Dataset

The Banana Ripeness Classification Dataset contains images grouped into four classes:

```text
train/
├── unripe/
├── ripe/
├── overripe/
└── rotten/

valid/
├── unripe/
├── ripe/
├── overripe/
└── rotten/

test/
├── unripe/
├── ripe/
├── overripe/
└── rotten/
```


---

## Project Workflow

```text
Banana Image
      ↓
Image Preprocessing
      ↓
EfficientNetB0 (Transfer Learning)
      ↓
Ripeness Classification
      ↓
Shelf-Life Estimation
```

---

## Data Preprocessing

The following preprocessing steps were applied:

* Image resizing to 224 × 224 pixels
* Data augmentation:

  * Random rotation
  * Zoom augmentation
  * Horizontal flipping
* EfficientNet preprocessing

```python
from tensorflow.keras.applications.efficientnet import preprocess_input
```

---

## Model Architecture

EfficientNetB0 was used as the feature extractor.

### Architecture

```text
Input Image (224×224×3)
          ↓
EfficientNetB0
          ↓
GlobalAveragePooling2D
          ↓
Dense(128, ReLU)
          ↓
Dropout(0.3)
          ↓
Dense(4, Softmax)
```

### Transfer Learning

EfficientNetB0 was frozen and only the classification head was trained.

```python
base_model.trainable = False
```

This preserved the pretrained ImageNet weights while learning banana-specific features.

---

## Training Configuration

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Image Size    | 224 × 224                |
| Batch Size    | 152                       |
| Optimizer     | Adam                     |
| Learning Rate | 0.0001                    |
| Loss Function | Categorical Crossentropy |
| Epochs        | 10                       |

---

## Model Performance

### Test Accuracy

```text
96.26%
```


---
## Shelf-Life Estimation

After classification, shelf life is estimated using the following mapping:

| Ripeness Stage | Estimated Shelf Life |
| -------------- | -------------------- |
| Unripe         | 6–9 days             |
| Ripe           | 3–5 days             |
| Overripe       | 1–2 days             |
| Rotten         | 0 days               |

Example:

```text
Predicted Class: Ripe
Estimated Shelf Life: 3–5 days
```

---

## Model Saving

```python
model.save("banana_shelf_life_model.keras")
```

Loading the model:

```python
from tensorflow.keras.models import load_model

model = load_model("banana_shelf_life_model.keras")
```

---

## Inference Example

```python
img = image.load_img(
    "banana.jpg",
    target_size=(224,224)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

prediction = model.predict(img_array)
```

---

## Technologies Used

* Python
* TensorFlow
* Keras
* EfficientNetB0
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Google Colab


## Conclusion

This project successfully demonstrates a deep learning-based banana freshness monitoring system using transfer learning. The EfficientNetB0 model achieved 96.26% test accuracy and effectively classified banana ripeness stages, enabling practical shelf-life estimation from a single image.
