import subprocess
import tkinter as tk
from tkinter import filedialog
from tkinter import *

import numpy
import torch
from PIL import ImageTk, Image, ImageDraw
from tensorflow.keras.models import load_model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'runs\train\exp\weights\last.pt')
model.eval()



# Các nhãn của các biển đã train
classes = {0: 'Stop',1:'camdoxe',2:'camvao',3:'camxedap'}

# Tạo giao diện
top = tk.Tk()
top.geometry('1000x1000')
top.title('Nhận diện biển báo GT')
top.configure(background='#008080')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def run_detection():
    detect_script = r'C:\Users\PC\PycharmProjects\pythonProject1\yolov5\detect.py'
    weights = r'C:\Users\PC\PycharmProjects\pythonProject1\yolov5\runs\train\exp\weights\last.pt'
    conf = 0.25

    command = f'python {detect_script} --source 0 --weights "{weights}" --conf {conf}'
    subprocess.call(command, shell=True)




#----------------------------------------------------------------------------------------------------
model = load_model(r'C:\Users\PC\PycharmProjects\pythonProject1\traffic_classifier1.h5')

# Các nhãn của các biển đã train
classes = {0: 'Tốc độ tối đa (20km/h)',
           1: 'Tốc độ tối đa (30km/h)',
           2: 'Tốc độ tối đa (50km/h)',
           3: 'Tốc độ tối đa (60km/h)',
           4: 'Tốc độ tối đa (70km/h)',
           5: 'Tốc độ tối đa (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Tốc độ tối đa (100km/h)',
           8: 'Tốc độ tối đa (120km/h)',
           9: 'Không được vượt',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Đường ưu tiên',
           13: 'Nhường đường',
           14: 'Dừng lại',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'Không vào   ',
           18: 'Cẩn thận',
           19: 'Chỗ ngoặt nguy hiểm vòng bên trái',
           20: 'Chỗ ngoặt nguy hiểm vòng bên phải',
           21: 'Double curve',
           22: 'Đường gập ghềnh',
           23: 'Đường trơn trượt',
           24: 'Road narrows on the right',
           25: 'Đường đang thi công',
           26: 'Biển báo giao thông',
           27: 'Pedestrians',
           28: 'Trẻ em qua đường',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Rẽ phải phía trước',
           34: 'Rẽ trái phía trước',
           35: 'Đi thẳng',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}

def classify(file_path):
    image = Image.open(file_path)
    image = image.convert('RGB')
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = image / 255.0
    print(image.shape)
    pred = model.predict(image)
    sign_class = numpy.argmax(pred, axis=1)[0]
    sign = classes[sign_class]
    print(sign)
    label.configure(foreground='#011638', text=sign)



def show_classify_button(file_path):
    classify_b = Button(top, text="Nhận diện", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.85)

#----------------------------------------------------------------------------------------------------

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


def stop_detection():
    global p
    if p is not None:
        # Dừng quá trình con nếu đang chạy
        p.terminate()
        p = None
        print("Camera stopped")
def show_additional_button():
    additional_b = Button(top, text="Camera", command=lambda: run_detection(), padx=10, pady=5)
    additional_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    additional_b.place(relx=0.79, rely=0.75)




upload = Button(top, text="Tải ảnh lên", command=upload_image, padx=10, pady=5)
upload.configure(background='#808000', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text="Nhận diện biển GT qua ảnh\n Nguyễn Hoàng Khánh Long-20110518 \n Hồ Hữu Quang Vinh-20110123", pady=20, font=('arial', 20, 'bold'))

heading.configure(background='#008080', foreground='#364156')
show_additional_button()
heading.pack()
top.mainloop()
