import cv2
import numpy as np
import os
os.environ['TF_ESTIMATOR_PUBLISH_ALL_METRICS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('traffic_classifier1.h5')

# Define the class names
classes = {0: 'Toc do toi da (20km/h)',
           1: 'Toc do toi da (30km/h)',
           2: 'Toc do toi da (50km/h)',
           3: 'Toc do toi da (60km/h)',
           4: 'Toc do toi da (70km/h)',
           5: 'Toc do toi da (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Toc do toi da (100km/h)',
           8: 'Toc do toi da (120km/h)',
           9: 'Khong duoc vuot',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Duong Uu Tien',
           13: 'Nhuong Duong',
           14: 'Dung Lai',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'Khong duoc vao   ',
           18: 'Can Than',
           19: 'Cho ngoat nguy hiem vong ben trai',
           20: 'Cho ngoat nguy hiem vong ben phai',
           21: 'Double curve',
           22: 'Duong gap ghenh',
           23: 'Duong tron truot',
           24: 'Road narrows on the right',
           25: 'Duong Dang Thi Cong',
           26: 'Bien Bao Giao Thong',
           27: 'Pedestrians',
           28: 'Tre Em Qua Duong',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Re Phai Phia Truoc',
           34: 'Re Phai Phia Truoc',
           35: 'Di Thang',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}

# Function to classify image
def classify_image(img):
    # Resize image to fit model input shape
    img = cv2.resize(img, (30, 30))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Thêm kênh màu RGB
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Make predictions using the model
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    return classes[class_id]

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Classify the image
    result = classify_image(gray)
    # Display the result on the original image
    cv2.putText(frame, result, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and exit
cap.release()
cv2.destroyAllWindows()
