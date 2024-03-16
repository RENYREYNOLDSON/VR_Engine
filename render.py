####################################### LOADING REQUIRED MODULES
from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector
import csv
import math
import numpy as np
import cv2
import random

####################################### SETUP OF THE ENGINE
#WINDOW
width = 500
height = 300
time=0

#LOADING MODELS
models=[]
for i in range(3):
	models.append(Model("data/headset.obj"))
for model in models:
	model.normalizeGeometry()

#CONSTANTS
ALPHA=0.5
GRAVITY = -9.8
DRAG_COEF=0.5
AIR_DENSITY=1.3
AREA=0.2
MASS=1

#VIEW INFORMATION
camera=[0,0,-10]
eye=np.array(camera)
orientation=[1,0,0,0]
lightColour=[200,240,240,255]#b,g,r

def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
	screenX = int((x+1)*width/2.0)
	screenY = int((y+1)*height/2.0)
	return screenX, screenY

def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal
	return normal / len(faceNormalsByVertex[vertIndex])
#####################################################################################

#######################################PROBLEM 1 - Perspective Projection and Transforms

class PerspectiveProjection():
	def __init__(self):
		#Save these in a class for better performance
		#Near Point
		l=-1.66666
		b=-1
		n=-2
		#Far Point
		r=1.666666#This value so same ratio as the screen
		t=1
		f=-10
		self.p=np.array([0,0,-1])
		#Perspective Transform
		Tp = np.array([[n,0,0,0],
					[0,n,0,0],
					[0,0,n+f,-f*n],
					[0,0,1,0]])

		Tst = np.array([[2/(r-l),0,0,-(r+l)/(r-l)],
						[0,2/(t-b),0,-(t+b)/(t-b)],
						[0,0,2/(n-f),-(n+f)/(n-f)],
						[0,0,0,1]])
		self.Tcan = Tst @ Tp


		#Viewport/Window Transform, scale to screen
		self.Tvp = np.array([[width/2,0,0,(width-1)/2],
						[0,height/2,0,(height-1)/2],
						[0,0,1,0],
						[0,0,0,1]])
					
	def getPerspectiveProjection1(self,x,y,z):#PROBLEM 1
		#View Transform QUESTION 1 VERSION OF CODE HERE
		up=np.array([0,1,0])
		c = np.divide((self.p-eye),np.linalg.norm(self.p-eye))#Check this
		zc=-c
		xc= np.cross(up,zc)
		yc= np.cross(zc,xc)

		I=np.array([[1,0,0,eye[0]],
					[0,1,0,eye[1]],
					[0,0,1,eye[2]],
					[0,0,0,1]])

		Teye=np.array([[xc[0],xc[1],xc[2],0],
					[yc[0],yc[1],yc[2],0],
					[zc[0],zc[1],zc[2],0],
					[0,0,0,1]])

		Teye=Teye @ I

		#Current Point
		P = np.array([[x],
					[y],
					[z],
					[1]])

		T = self.Tvp @ self.Tcan @ Teye#MAIN TRANSFORM
		pT = T @ P
		if pT[3]==0:
			return 0,0
		else:
			screenX=int(pT[0]/pT[3])
			screenY=int(pT[1]/pT[3])
		return screenX,screenY



	def getPerspectiveProjection3(self,x,y,z):#PROBLEM 3, INCLUDES TRACKING AND DISTORTION
		I=np.array([[1,0,0,eye[0]],
					[0,1,0,eye[1]],
					[0,0,1,eye[2]],
					[0,0,0,1]])

		q=conjugateQuaternion(orientation)
		ReyeI=quaternionToMatrix(q)
		#Reye matrix version of orientation
		#Get cojuagte and expand to numpy array
		Teye=np.pad(ReyeI,(0,1),mode="constant")
		Teye[3][3]=1#Add identity value
		Teye=Teye@I
		
		#Current Point
		P = np.array([[x],
					[y],
					[z],
					[1]])
		#Combining the Transforms
		T = self.Tcan @ Teye
		pT=T @ P
		if pT[3]==0:
			return 0,0
		else:
			x,y=distortionCorrection(pT[0]/pT[3],pT[1]/pT[3])
			x=int(width/2+x*width/2)
			y=int(height/2+y*height/2)
		return x,y

cam = PerspectiveProjection()

##TRANSFORMS 
def transform(x2,y2,z2,alpha,beta,gamma,scale):
	#Yaw
	Rz=np.array([[math.cos(gamma),-math.sin(gamma),0,0],
				 [math.sin(gamma),math.cos(gamma),0,0],
				 [0,0,1,0],
				 [0,0,0,1]])
	#Pitch
	Rx=np.array([[1,0,0,0],
				 [0,math.cos(beta),-math.sin(beta),0],
				 [0,math.sin(beta),math.cos(beta),0],
				 [0,0,0,1]])
	#Roll
	Ry=np.array([[math.cos(alpha),0,math.sin(alpha),0],
				 [0,1,0,0],
				 [-math.sin(alpha),0,math.cos(alpha),0],
				 [0,0,0,1]])
	#Scale
	Rscale=np.array([[scale,0,0,0],
					 [0,scale,0,0],
					 [0,0,scale,0],
					 [0,0,0,1]])

	#Add Translation
	Trb = Rscale @ Ry @ Rx @ Rz
	Trb[0][3]=x2#CHANGE ALL OF THIS AROUND TO M. Like in lecture 2
	Trb[1][3]=y2
	Trb[2][3]=z2
	return Trb

def transformModel(Trb,model):
	#Transform Model Here
	for p in model.vertices:
		v=np.array([[p.x],
					[p.y],
					[p.z],
					[1]])
		v2 = Trb @ v
		p.x=v2[0][0]
		p.y=v2[1][0]
		p.z=v2[2][0]

	faceNormals = {}
	for face in model.faces:
		p0, p1, p2 = [model.vertices[i] for i in face]
		faceNormal = (p2-p0).cross(p1-p0).normalize()
		for i in face:
			if not i in faceNormals:
				faceNormals[i] = []
			faceNormals[i].append(faceNormal)
	# Calculate vertex normals
	vertexNormals = []
	for vertIndex in range(len(model.vertices)):
		vertNorm = getVertexNormal(vertIndex, faceNormals)
		vertexNormals.append(vertNorm)
	return vertexNormals
#####################################################################################

#######################################PROBLEM 2 - IMU Data and Quaternions
class IMURecord():
	def __init__(self,record):
		self.t=record[0]
		self.gx,self.gy,self.gz=math.radians(record[1]),math.radians(record[2]),math.radians(record[3])#Convert this to radians
		norm=math.sqrt(record[4]**2+record[5]**2+record[6]**2)
		self.ax=record[4]/norm
		self.ay=record[5]/norm
		self.az=record[6]/norm
		#Normalize these, careful with NaN divisions
		norm=math.sqrt(record[7]**2+record[8]**2+record[9]**2)
		self.mx=record[7]/norm
		self.my=record[8]/norm
		self.mz=record[9]/norm
		
	def printInfo(self):
		print("Time: "+str(self.t)+"\nGyroscope: x:"+str(self.gx)+" y:"+str(self.gy)+" z:"+str(self.gz))
		print("Accelerometer: x:"+str(self.ax)+" y:"+str(self.ay)+" z:"+str(self.az))
		print("Magnetometer: x:"+str(self.mx)+" y:"+str(self.my)+" z:"+str(self.mz))

	def getG(self):#Extend to accelerometer data
		mult=1
		axis = np.array([self.gx,self.gy,self.gz])/math.sqrt(self.gx**2*mult+self.gy**2*mult+self.gz**2)*mult
		rate = math.sqrt(self.gx**2+self.gy**2+self.gz**2)*(1/256)*mult
		q=[math.cos(rate/2),axis[0]*math.sin(rate/2),axis[1]*math.sin(rate/2),axis[2]*math.sin(rate/2)]
		return q

	def getA(self):#Extend to accelerometer data
		mult=10
		axis = np.array([self.ax,self.ay,self.az])/math.sqrt(self.ax**2*mult+self.ay**2*mult+self.az**2)*mult
		rate = math.sqrt(self.ax**2+self.ay**2+self.az**2)*(1/256)*mult
		q=[math.cos(rate/2),axis[0]*math.sin(rate/2),axis[1]*math.sin(rate/2),axis[2]*math.sin(rate/2)]
		return q


IMUData=[]
with open("IMUData.csv") as file:
	reader=csv.reader(file,delimiter=",")
	next(reader,None)
	for row in reader:
			IMUData.append(IMURecord([float(x) for x in row]))
file.close()
IMUData[1].printInfo()#Here to check it's loaded


def toQuaternion(e):#CONVERTS EULER TO QUATERNION
	yaw=e[0]/2
	pitch=e[1]/2
	roll=e[2]/2
	w = math.cos(yaw)*math.cos(pitch)*math.cos(roll)-math.sin(yaw)*math.sin(pitch)*math.sin(roll)
	x = math.sin(yaw)*math.cos(pitch)*math.cos(roll)+math.cos(yaw)*math.sin(pitch)*math.sin(roll)
	y = math.cos(yaw)*math.sin(pitch)*math.cos(roll)-math.sin(yaw)*math.cos(pitch)*math.sin(roll)
	z = math.cos(yaw)*math.cos(pitch)*math.sin(roll)+math.sin(yaw)*math.sin(pitch)*math.cos(roll)
	return [w,x,y,z]

def toEuler(q):#CONVERTS QUATERNION [w0,x1,y2,z3] TO AN EULER [x,y,z]
	alpha = math.atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]**2+q[2]**2))
	val = 2*(q[0]*q[2]-q[3]*q[1])
	if val>=1:
		beta=math.pi/2
	elif val<=-1:
		beta=-math.pi/2
	else:
		beta = math.asin(val)
	gamma = math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
	return [alpha,beta,gamma]

def conjugateQuaternion(q):#GETS CONJUGATE OF A QUATERNION
	return [q[0],-q[1],-q[2],-q[3]]

def productQuaternion(q1,q2):#GETS PRODUCTS OF 2 QUATERNIONS
	w = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
	x = q1[0]*q2[1]+q1[1]*q2[0]+q1[2]*q2[3]-q1[3]*q2[2]
	y = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
	z = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
	return [w,x,y,z]#Correct

#For Dead Reckoning Reye
def quaternionToMatrix(q):#CHANGES A QUARTERNION TO MATRIX
	a,b,c,d=q
	row1=[2*(a**2+b**2)-1,2*(b*c-a*d),2*(b*d+a*c)]
	row2=[2*(b*c+a*d),2*(a**2+c**2)-1,2*(c*d-a*b)]
	row3=[2*(b*d-a*c),2*(c*d+a*b),2*(a**2+d**2)-1]
	return np.array([row1,row2,row3])#Correct



#####################################################################################

#######################################PROBLEM 3 - Dead Reckoning Filter
def deadReckoningFilter():

	#Rotate accelerometer output by a= q(t)-1 * a * q(t) to get it into the global frame 
	a = IMUData[time].getA()

	a = productQuaternion(productQuaternion(conjugateQuaternion(orientation),a),orientation)

	#Get new acceleration quaternion to use in tracking
	tiltAxis = [a[3],0,a[1]]
	tiltError = ALPHA*math.acos((a[1]*0+a[2]*1+a[3]*0)/(math.sqrt(a[1]**2+a[2]**2+a[3]**2)*math.sqrt(0**2+1**2+0**2)))#Dot over modulus a.b = 

	q=[math.cos(tiltError/2),tiltAxis[0]*math.sin(tiltError/2),tiltAxis[1]*math.sin(tiltError/2),tiltAxis[2]*math.sin(tiltError/2)]

	#Complementary Filter
	Og = productQuaternion(IMUData[time].getG(),orientation)#Gyroscope Estimate
	Oa = productQuaternion(q,orientation)#Accelerometer Estimate

	O = productQuaternion(Oa,Og)#Not correct in this

	return Og
#####################################################################################

#######################################PROBLEM 4 - Pre Distortion Correction
def distortionCorrection(x,y):
	r = math.sqrt(x**2+y**2)
	theta = math.atan2(y,x)
	c1=0.5
	c2=0
	rd = r + c1*r**3 + c2*r**5
	x = rd * math.cos(theta) 
	y = rd * math.sin(theta) 
	return x,y
#####################################################################################

#######################################PROBLEM 5 - Gravity and Physics Implementation
def drag(x):
	return DRAG_COEF*((AIR_DENSITY*x**2)/2)*AREA

def applyPhysics(v):
	weight = MASS * GRAVITY
	acceleration = (weight-drag(v[1]))/MASS
	v[0]-=drag(v[0])/256
	v[1]+=acceleration/256
	v[2]-=drag(v[2])/256
	return v

def checkCollisions():
	for model1 in models:
		for model2 in models:
			if model1!=model2:
				p1=model1.vertices[0]
				p2=model2.vertices[0]
				dist=abs(math.sqrt(p1.x**2+p1.y**2+p1.z**2)-math.sqrt(p2.x**2+p2.y**2+p2.z**2))
				p1=np.array([p1.x,p1.y,p1.z])
				p2=np.array([p2.x,p2.y,p2.z])
				radius=0.2
				if dist<radius:
					#Use newtons law of motion to do this!
					v1=np.array(model1.velocity)
					v2=np.array(model2.velocity)

					#Calculate new velocities by using normals vectors
					n=(p1-p2)/np.linalg.norm(p1-p2)
					vrel=v1-v2
					vnorm=np.array(vrel[0]*n[0]+vrel[1]*n[1]+vrel[2]*n[2])*n

					model1.new_velocity=v1+vnorm
					model2.new_velocity=v2-vnorm

	for model in models:
		if len(model.new_velocity)>0:
			model.velocity=model.new_velocity
			model.new_velocity=[]
#####################################################################################


for model in models[2:]:
	Trb=transform(random.uniform(-5,5),random.uniform(1,5),random.randint(-1,3),0,0,0,0.4)
	model.normals=transformModel(Trb,model)

#TESTING COLLISIONS
Trb1=transform(3,3,-0.5,0,0,0,0.4)
models[0].normals=transformModel(Trb1,models[0])
models[0].velocity=[-0.1,0,0]
models[0].a_velocity=[0.1,0.05,0]

Trb2=transform(-3,-3,0,0,0,0,0.4)
models[1].normals=transformModel(Trb2,models[1])
models[1].velocity=[0.05,0.05,0]
models[1].a_velocity=[0.05,0,0]



while True:#MAIN WINDOW LOOP
	orientation=deadReckoningFilter()
	zBuffer = [-float('inf')] * width * height
	image = Image(width, height, Color(255, 150, 0, 255))
	# Render the image iterating through faces
	for model in models:
		#Gravity here
		model.velocity=applyPhysics(model.velocity)
		#TRANSFORM MODELS HERE!
		Trb=transform(model.velocity[0],model.velocity[1],model.velocity[2],model.a_velocity[0],model.a_velocity[1],model.a_velocity[2],1)
		model.normals=transformModel(Trb,model)
		#Use this for falling objects
		if model.vertices[0].y<-4:
			Trb=transform(random.uniform(-5,5),5,random.uniform(-1,3),random.uniform(0,5),random.uniform(0,5),random.uniform(0,5),1)
			model.velocity=[0,0,0]
			model.normals=transformModel(Trb,model)
	checkCollisions()#Check object collisions!
	for model in models:
		for face in model.faces:
			p0, p1, p2 = [model.vertices[i] for i in face]
			n0, n1, n2 = [model.normals[i] for i in face]

			# Define the light direction
			lightDir = Vector(0, 0, -1)

			# Set to true if face should be culled
			cull = False
			# Transform vertices and calculate lighting intensity per vertex
			transformedPoints = []
			for p, n in zip([p0, p1, p2], [n0, n1, n2]):
				intensity = n * lightDir
				# Intensity < 0 means light is shining through the back of the face
				# In this case, don't draw the face at all ("back-face culling")
				if intensity < 0:
					cull = True
					break

				#screenX, screenY = getOrthographicProjection(p.x, p.y, p.z)
				#screenX,screenY=cam.getPerspectiveProjection1(p.x,p.y,p.z)
				screenX,screenY=cam.getPerspectiveProjection3(p.x,p.y,p.z)
				transformedPoints.append(Point(screenX, screenY, p.z, Color(intensity*lightColour[0], intensity*lightColour[1], intensity*lightColour[2], lightColour[3])))

			if not cull:
				Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw(image, zBuffer)
				pass
	image.render()
	#image.saveAsPNG(str(time)+".png")
	cv2.waitKey(1)
	time+=1
	print(time)


