import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model


y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    image = np.array(image)
    image = image / 255.0
    data.append(image)

X_test = np.array(data)
model=load_model('traffic_classifier1.h5')
# Dự đoán xác suất của các lớp
pred_prob = model.predict(X_test)

# Lấy lớp có xác suất cao nhất làm kết quả dự đoán
pred = np.argmax(pred_prob, axis=1)
# Độ chính xác của dữ liệu thử nghiệm
from sklearn.metrics import accuracy_score

print(accuracy_score(labels, pred))