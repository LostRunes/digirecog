# Handwritten Digit Recognition Using Machine Learning

Recognises handwritten digits (0-9) from the MNIST dataset and from digits drawn by the user.
Four classical ML baselines are compared against a Convolutional Neural Network (CNN).

## Folder layout
```
src/data.py              load MNIST, normalise, train/val/test split (55k / 5k / 10k)
src/preprocess.py        convert any drawn/scanned digit to MNIST format (crop, 20x20 resize, centre of mass)
src/train_classical.py   Logistic Regression, Random Forest, k-NN, SVM
src/train_cnn.py         CNN with augmentation, BatchNorm, Dropout
src/evaluate.py          confusion matrix, curves, comparison charts, classification report
src/app.py               Tkinter drawing pad with live prediction
src/predict.py           predict from an image file
results/                 metrics (json) and figures (png)
models/                  saved models
alert.py                 beeps until Ctrl+Space (used to get the user's attention)
make_ppt.py              builds the presentation
```

## Run
```
pip install -r requirements.txt
cd src
python train_classical.py
python train_cnn.py
python evaluate.py
python app.py                 # draw digits
python predict.py my_digit.png
```
