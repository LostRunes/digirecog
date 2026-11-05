# Handwritten Digit (and Letter) Recognition Using Machine Learning

Recognises handwritten digits (0-9) from the MNIST dataset and from digits drawn by the user.
Four classical ML baselines are compared against a Convolutional Neural Network (CNN).

## Folder layout
```
src/data.py              load MNIST, normalise, train/val/test split (55k / 5k / 10k)
src/preprocess.py        convert any drawn/scanned digit to MNIST format (crop, 20x20 resize, centre of mass)
src/train_classical.py   Logistic Regression, Random Forest, k-NN, SVM
src/train_cnn.py         CNN with augmentation, BatchNorm, Dropout (digits, 10 classes)
src/prepare_emnist.py    builds the 36-class (0-9, A-Z) dataset from the EMNIST zip
src/train_alnum.py       trains the 36-class CNN
src/evaluate_alnum.py    metrics and figures for the 36-class model
src/evaluate.py          confusion matrix, curves, comparison charts, classification report
src/app.py               Tkinter drawing pad with live prediction
src/predict.py           predict from an image file
results/                 metrics (json) and figures (png)
models/                  saved models
alert.py                 beeps until Ctrl+Space (used to get the user's attention)
make_ppt.py              builds the presentation
```

## Results (test sets never used for training)
| Model | Data | Accuracy |
|---|---|---|
| Logistic Regression / Random Forest / k-NN / SVM | MNIST digits | 92.14 / 96.91 / 97.62 / 98.61 % |
| CNN (10 classes) | MNIST digits | **99.34 %** |
| CNN (36 classes, 0-9 + A-Z) | EMNIST | 87.57 % (look-alikes such as L/1, O/0, I/1 cause most errors) |

## Run
```
pip install -r requirements.txt
python src/app.py                           # draw digits / letters (switch mode at the top)
python src/predict.py my_img.png            # digit model
python src/predict.py my_img.png letters    # 36-class model

# optional: retrain from scratch
python src/train_classical.py
python src/train_cnn.py                     # ~25 min on CPU
python src/prepare_emnist.py archive.zip    # EMNIST zip (Kaggle 'crawford/emnist' or NIST gzip.zip)
python src/train_alnum.py                   # ~25 min on CPU
python src/evaluate.py && python src/evaluate_alnum.py
python make_ppt.py
```
Scripts work from any folder. The trained models are included, so the app runs without retraining.
