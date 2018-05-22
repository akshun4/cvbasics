import cv2
import numpy as np
from matplotlib import pyplot as plt

img1=cv2.imread("Input/image1.jpg",1)				#image1 read in color
img0=cv2.imread("Input/image1.jpg",0)				#image1 read in grayscale
img11 = cv2.imread('Input/openCVLogo.png')

def a1grey():
	gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	cv2.imshow("Converted Image",gray)
	cv2.imwrite("Output/grayedimage.jpg",gray)

def a1hsv():
	hsv=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
	cv2.imshow("Converted Image",hsv)
	cv2.imwrite("Output/HSVimage.jpg",hsv)

def a1rgb():
	rgb=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
	cv2.imshow("Converted Image",rgb)
	cv2.imwrite("Output/RGBimage.jpg",rgb)

def lstfunc():
	functions=dir(cv2)
	for fun in functions:
	    print(fun)
	    print("\n")

def a2accpix():
	px=img1[100,100]
	print("The pixel values of the coordinate [100,100] is ",px)

	print("The blue pixel values of the coordinate [100,100] is ",px[0])
	print("The green pixel values of the coordinate [100,100] is ",px[1])
	print("The red pixel values of the coordinate [100,100] is ",px[2])

def a2modpix():
	px=img1[100,100]
	print("The pixel values of the coordinate [100,100] is ",px)
	px=[245,100,234]

	print(" \nThe modified pixel value at coordinate [100,100] is  ",px)

	px[0]=245
	print(" \nThe (only blue) modified pixel value at coordinate [100,100] is  ",img1[100,100,0])
	px[1]=230
	print(" \nThe (only green) modified pixel value at coordinate [100,100] is  ",img1[100,100,1])
	px[2]=78
	print(" \nThe (only red) modified pixel value at coordinate [100,100] is  ",img1[100,100,2])

def a3accshapecolor():
	rows,columns,channels=img1.shape
	dimension=img1.shape
	print("The dimension of an image is ",dimension)
	print("The no of rows in an image is ", rows)
	print("The no of columns in an image is ",columns)
	print("The channels in an image is ",channels)

def a3accshapegray():
	rows,columns=img0.shape
	dimension=img0.shape

	print("The dimension of an image is ",dimension)
	print("The no of rows in an image is ", rows)
	print("The no of columns in an image is ",columns)

def borderplot():
	BLUE=[255,0,0]
	replicate = cv2.copyMakeBorder(img11,10,10,10,10,cv2.BORDER_REPLICATE)
	reflect = cv2.copyMakeBorder(img11,10,10,10,10,cv2.BORDER_REFLECT)
	reflect101 = cv2.copyMakeBorder(img11,10,10,10,10,cv2.BORDER_REFLECT_101)
	wrap = cv2.copyMakeBorder(img11,10,10,10,10,cv2.BORDER_WRAP)
	constant= cv2.copyMakeBorder(img11,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

	plt.subplot(231),plt.imshow(img11,'gray'),plt.title('ORIGINAL')
	plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
	plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
	plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
	plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
	plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

	plt.show()

def changeFormat():
	cv2.imwrite("Output/image1.png",img1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def destroyPresseda():
	cv2.imshow("Image",img0)
	k=cv2.waitKey(0)
	if k== ord('a'):
		cv2.destroyAllWindows()

def fsaveclose():
	cv2.imshow("Imagef",img1)
	k = cv2.waitKey(0)
	if k == 27: 
		cv2.destroyAllWindows()
	elif k== ord('s'): # Wait for s key to save and exit
	    cv2.imwrite("Output/image23.png",img1)
	    cv2.destroyAllWindows()

def l1loadNamedResize():
	cv2.namedWindow("FIRSTIMAGE",cv2.WINDOW_NORMAL)
	cv2.imshow("FIRSTIMAGE",img1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def l2loadNamed():
	cv2.namedWindow("Secondimage",cv2.WINDOW_AUTOSIZE)
	cv2.imshow("Secondimage",img1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def o1gr2rgb():
	rgb=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
	cv2.imshow("DISPLAYING IMAGE IN RGB FORMAT",rgb)
	cv2.imwrite("Output/BGR2RGB.jpg",rgb)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def o2bgr2hsv():
	hsv=cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
	cv2.imshow("DISPLAYING IMAGE IN HSV FORMAT",hsv)
	cv2.imwrite("Output/BGR2HSV.jpg",hsv)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def t1Line():
	line=cv2.line(img1,(0,0),(511,511),(255,0,0),5)
	cv2.imshow("Display",line)
	cv2.imwrite("Output/LINEONIMAGE.jpg",line)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def t2Rct():
	img1=cv2.imread("Input/image1.jpg",1)
	rect=cv2.rectangle(img1,(384,0),(510,128),(0,255,0),3)
	cv2.imshow("Display2",rect)
	cv2.imwrite("Output/RECTONIMAGE.jpg",rect)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def t2circle():
	img1=cv2.imread("Input/image1.jpg",1)
	circle=cv2.circle(img1,(447,63), 63, (0,0,255), -1)
	cv2.imshow("Display3",circle)
	cv2.imwrite("Output/CIRCLEONIMAGE.jpg",circle)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def t3circlewithThick():
	img1=cv2.imread("Input/image1.jpg",1)
	circle=cv2.circle(img1,(447,63), 63, (0,0,255), 2)
	cv2.imshow("Display4",circle)
	cv2.imwrite("Output/CIRCLEWITHTHICKNESS.jpg",circle)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def t4ellipse():
	img1=cv2.imread("Input/image1.jpg",1)
	el=cv2.ellipse(img1,(256,256),(100,50),0,0,180,255,-1)
	cv2.imshow("Display5",el)
	cv2.imwrite("Output/ELLIPSEONIMAGE.jpg",el)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def t5text():
	img1=cv2.imread("Input/image1.jpg",1)
	font = cv2.FONT_HERSHEY_SIMPLEX
	txt=cv2.putText(img1,'OpenCV',(20,500), font, 4,(201,220,220),8,cv2.LINE_AA)
	cv2.imshow("Display6",txt)
	cv2.imwrite("Output/TEXTONIMAGE.jpg",txt)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def vid1play():
	cap = cv2.VideoCapture(0)
	fourcc = cv2.VideoWriter_fourcc(*'MJPG') #The fourcc is used to mention the video codec for encoding the video
	#The VideoWriter() accepts four parameters.These are as follows:
	    # 1) The output file name along with the format in which you want to save the file
	    # 2) The video codec or fourcc.
	    # 3) Frames per second which is used to mention how fast or slow your video should play along with the frame size that is the size of the window
	    # 4) isColor flag. If it is True, encoder expect color frame, otherwise it works with grayscale frame.
	out = cv2.VideoWriter('Output/webcam.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))), isColor=True)
	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret==True:
			out.write(frame)
			cv2.imshow('frame',frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	# When everything done, release the capture
	cap.release()
	out.release()
	# Destroy all the previously created window objects.
	cv2.destroyAllWindows()

def vid2diffFormat():
	cap=cv2.VideoCapture("Input/Countdown.mp4")# COSTA.mp4 is the filename of the video along with the format name
	fourcc = cv2.VideoWriter_fourcc(*'MJPG') # For fourcc visit fourcc.org

	#using isColor = False to save gray scale video
	out=cv2.VideoWriter('Output/ConvertedVideo.avi', fourcc, 20, (int(cap.get(3)), int(cap.get(4))), isColor=False)
	    
	#Step 3: Return the video frame by frame
	while(cap.isOpened()):
	    ret ,frame=cap.read()
	    if ret==True:
	        #Step 4:- Displaying it to the user after performing color inversion
	        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	        
	        #Step 5:- Write the file into the disk with specific format
	        out.write(gray)
	        cv2.imshow("DIPLAYING TO USER",gray)
	        if cv2.waitKey(25) & 0xFF==ord('a'):
	            break
	    else:
	        break
	# Step 6 : Release everything after use
	cap.release()
	out.release()
	cv2.destroyAllWindows()

