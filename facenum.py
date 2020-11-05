import cv2
import matplotlib.pyplot as plt

# 载入图片
p_color = cv2.imread('../face1.jpg')
# 将图像灰度化
p_gray = cv2.cvtColor(p_color,cv2.COLOR_BGR2GRAY)
#展示灰度图像
plt.imshow(p_gray,cmap='gray')
plt.show()

# 载入haarcascade级联分类器cascade classifier
haarcascade = cv2.CascadeClassifier('../haarcascade_frontalface_alt.xml')
# 检测图像人脸个数
faces = haarcascade.detectMultiScale(p_gray, scaleFactor=1.1, minNeighbors=5)
print("人脸的个数：",len(faces))

# 给检测到的人脸加矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(p_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
# 展示检测到的人脸并加矩形框
plt.imshow(cv2.cvtColor(p_color, cv2.COLOR_BGR2RGB))
plt.show()