#Import packages:
import scipy.ndimage
import sklearn; sklearn.show_versions()
import numpy
from numpy import shape
import cv2
from skimage.segmentation import watershed
import skimage.feature.peak
import imutils



#Define Variables:
filename = 'cell_images/' #enter file name here
img_name = filename.replace('cell_images/', '')
img = cv2.imread(filename,0) #imread keys: -1: normal, 0:Greyscale, 1:ignore transparency

#load base image (for troubleshooting purposes):
'''
cv2.resize(img, None, fx=0.1, fy=0.1)
cv2.imshow(img_name,img)
cv2.waitKey(0)
cv2.destroyWindow() #Pressing any key will destroy window. Prevents unnessessary windows from staying open
'''

#Define base color and color thresholds for image detection. uncomment next line isntead of this one if you want to save image in color:
#grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grey = img 
detection_threshold = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#cv2.imshow(img_name,detection_threshold) #uncomment for troubleshooting


#scipy multidimentional image processing:
distance_transform_img = scipy.ndimage.distance_transform_edt(detection_threshold)
local_max = skimage.feature.peak_local_max(distance_transform_img, indices=False, min_distance=20,labels=detection_threshold)
cv2.imshow("Transformed image" + img_name, img)


#making markers and labels:
marker = scipy.ndimage.label(local_max, structure=numpy.ones((3, 3)))[0]
labels = watershed(-distance_transform_img,marker,mask=detection_threshold)

print("Unique labels found:".format(len(numpy.unique(labels))-1))


cell_count = 0

for label in numpy.unique(labels):
    if label == 0:
        continue
    mask = numpy.zeros(grey.shape, dtype="uint8")
    mask[label == labels] = 225


    contours = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    grabbed_contours = imutils.grab_contours(contours)
    c = max(grabbed_contours, key=cv2.contourArea)

#placing designator over label
    ((x,y),r) = cv2.minEnclosingCircle(c)
    cv2.circle(img,(int(x),int(y)),int(r),(0,255,0),2)
    cv2.putText(img,"#{}".format(label),(int(x)-10,int(y)),fontFace=cv2.QT_FONT_NORMAL,fontScale=0.5,color=(0,0,225))
    cell_count +=1 

#loading other images
cv2.imshow("Transformed image: " + img_name, cv2.resize(img, None, fx=0.5, fy=0.5))
#cv2.resize(img, None, fx=0.02, fy=0.02)
cv2.waitKey(0)
cv2.destroyWindow("Transformed image: " + img_name)

cv2.imwrite('output_images/'+img_name +"_count="+str(cell_count)+".jpg",img)
print("total cells:" + str(cell_count))
#cv2.waitKey(0)