import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import cv2

'''

data = pd.read_csv(file)

x = np.array(data.loc[:,'x'])
z = np.array(data.loc[:,'z'])

xref = np.linspace(x[0],x[-1],20)
zref = interp1d(x,z,kind='quadratic')(xref)

data_ref = pd.DataFrame(np.vstack((xref,zref)).T, columns=['x','z'])
delta1_df.to_csv(file[-4]+'_ref.csv',index=False,sep=',',decimal='.')

'''

file = r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NPU_LSC_7613\Geometry\NPU-LSC-7613_.png'
file_2 = r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NPU_LSC_7613\Geometry\NPU-LSC-7613_exp.png'


img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

#convert img to grey
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#set a thresh
thresh = 100
#get threshold image
ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = np.zeros(img.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
#save image
cv2.imwrite(file_2,img_contours)
'''

# Reading image
font = cv2.FONT_HERSHEY_COMPLEX
img2 = cv2.imread(file, cv2.IMREAD_COLOR)

# Reading same image in another
# variable and converting to gray scale.
img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

# Converting image to a binary image
# ( black and white only image).
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

# Detecting contours in image.
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)

# Going through every contours found in the image.
for cnt in contours:

    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

    # draws boundary of contours.
    cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)

    # Used to flatted the array containing
    # the co-ordinates of the vertices.
    n = approx.ravel()
    i = 0

    for j in n:
        if (i % 2 == 0):
            x = n[i]
            y = n[i + 1]

            # String containing the co-ordinates.
            string = str(x) + " " + str(y)

            if (i == 0):
                # text on topmost co-ordinate.
                cv2.putText(img2, "Arrow tip", (x, y),
                            font, 0.5, (255, 0, 0))
            else:
                # text on remaining co-ordinates.
                cv2.putText(img2, string, (x, y),
                            font, 0.5, (0, 255, 0))
        i = i + 1
    print()

# Showing the final image.
cv2.imshow('image2', img2)

# Exiting the window if 'q' is pressed on the keyboard.
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
'''