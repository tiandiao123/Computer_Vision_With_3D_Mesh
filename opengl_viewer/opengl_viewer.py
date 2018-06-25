from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from PIL import Image

from .camera import Camera

import trimesh 
import numpy as np

# read in ply
mesh = trimesh.load_mesh('PLY/ASR.ply')
vertices = mesh.vertices.astype(np.float32)
colors = mesh.vertex_normals.astype(np.float32)
indices = mesh.faces.astype(np.uint32)
width=300
height=300

# read in texture image
I = Image.open('sampled_texture.jpg')
tw, th = I.size
Idata = I.tobytes("raw", "RGB", 0, -1)

def draw_mesh():
    #glBindTexture( GL_TEXTURE_2D, glGenTextures(1)) 
    glBegin(GL_TRIANGLES)
    for i in range(0, len(indices)):
        index=indices[i][0]
	#glTexCoord2f(0.0, 0.0)
	r,g,b = I.getpixel((index%tw, index%th))
        glColor3f(*[r/255.0, g/255.0, b/255.0])
        glNormal3f(*[colors[index][0], colors[index][1], colors[index][2]])
        glVertex3f(*[vertices[index][0], vertices[index][1], vertices[index][2]])

        index=indices[i][1]
	#glTexCoord2f(0.0, 1.0)
	r,g,b = I.getpixel((index%tw, index%th))
        glColor3f(*[r/255.0, g/255.0, b/255.0])
        glNormal3f(*[colors[index][0], colors[index][1], colors[index][2]])
        glVertex3f(*[vertices[index][0], vertices[index][1], vertices[index][2]])

        index=indices[i][2]
	#glTexCoord2f(1.0, 0.0)
	r,g,b = I.getpixel((index%tw, index%th))
        glColor3f(*[r/255.0, g/255.0, b/255.0])
        glNormal3f(*[colors[index][0], colors[index][1], colors[index][2]])
        glVertex3f(*[vertices[index][0], vertices[index][1], vertices[index][2]])
    glEnd()
    #glBindTexture( GL_TEXTURE_2D, 0)

class OpenGlViewer:
    def __init__(self):
        self.title = "Simulation" #title
        self.width = width
        self.height = height

	self.camera = Camera()
	self.step_size = 0.05

    def draw(self):
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
	gluLookAt(self.camera.position[0], self.camera.position[1], self.camera.position[2], self.camera.position[0] + self.camera.forward[0], self.camera.position[1] + self.camera.forward[1], self.camera.position[2] + self.camera.forward[2], self.camera.up[0], self.camera.up[1], self.camera.up[2])
        
	lightAmbient = np.array([0.5, 0.5, 0.5, 1])
	lightDiffuse = np.array([0.5, 0.5, 0.5, 1])
	lightSpecular = np.array([0.1, 0.1, 0.1, 1])
	lIdentity = np.identity(4)
	modelview = glGetDoublev( GL_MODELVIEW_MATRIX )
	lPosition = self.camera.position
	lDirection = self.camera.forward

	glEnable( GL_LIGHTING )
	glLightModelfv( GL_LIGHT_MODEL_AMBIENT, lightAmbient )
	#glLightModelfv( GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE )
	glLightModelfv( GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE )
	glLightfv ( GL_LIGHT0, GL_AMBIENT, lightAmbient )
	glLightfv ( GL_LIGHT0, GL_DIFFUSE, lightDiffuse )
	glLightfv ( GL_LIGHT0, GL_SPECULAR, lightSpecular )
	glLightfv ( GL_LIGHT0, GL_POSITION, lPosition )
	glLightfv ( GL_LIGHT0, GL_SPOT_DIRECTION, lDirection )
	#glLightfv ( GL_LIGHT0, GL_SPOT_CUTOFF, 10.0)
	#glLightfv ( GL_LIGHT0, GL_SPOT_EXPONENT, 0.1)
	glEnable ( GL_LIGHT0 )
	#glDisable( GL_LIGHT0 )
	glDisable( GL_LIGHT1 )
	glDisable( GL_LIGHT2 )
	glDisable( GL_LIGHT3 )
	glDisable( GL_LIGHT4 )
	glDisable( GL_LIGHT5 )
	glDisable( GL_LIGHT6 )
	glDisable( GL_LIGHT7 )

#        vp = glGetIntegerv(GL_VIEWPORT)
#        glRenderMode(GL_SELECT)
#        gluPickMatrix(0, 0, width, height, vp)
#        gluPerspective(45.0, float(width)/float(height), 0.0001, 50.0)
#        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )

        self.create_vbo()
        self.draw_vbo
	draw_mesh()

        glFlush()
        glutSwapBuffers()


    def create_vbo(self):
	buffers = glGenBuffers(3)
    	glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
    	glBufferData(GL_ARRAY_BUFFER, 
            len(vertices)*3,  # byte size
            (ctypes.c_float*len(vertices.flat))(*vertices.flat), 
            GL_STATIC_DRAW)
    	glBindBuffer(GL_ARRAY_BUFFER, buffers[1])
    	glBufferData(GL_ARRAY_BUFFER, 
            len(colors)*3, # byte size 
            (ctypes.c_float*len(colors.flat))(*colors.flat), 
            GL_STATIC_DRAW)
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2])
    	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
            len(indices)*3, # byte size
            (ctypes.c_uint*len(indices.flat))(*indices.flat), 
            GL_STATIC_DRAW)

    def draw_vbo():
    	glEnableClientState(GL_VERTEX_ARRAY)
	glEnableClientState(GL_NORMAL_ARRAY)
    	glEnableClientState(GL_COLOR_ARRAY)
    	glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
    	glVertexPointer(3, GL_FLOAT, 0, None+0*len(vertices)*(ctypes.c_float*len(vertices.flat))(*vertices.flat))
	glNormalPointer(   GL_FLOAT, 0, None+1*len(colors)*(ctypes.c_float*len(colors.flat))(*colors.flat))
    	glColorPointer (3, GL_FLOAT, 0, None+2*len(colors)*(ctypes.c_float*len(colors.flat))(*colors.flat))

    	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[1])
    	glDrawElements(GL_TRIANGLES, len(indices)*3, GL_UNSIGNED_INT, None)
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    	glDisableClientState(GL_COLOR_ARRAY)
	glDisableClientState(GL_NORMAL_ARRAY)
    	glDisableClientState(GL_VERTEX_ARRAY)
	glBindBuffer(GL_ARRAY_BUFFER, 0)

    def view(self):
        '''
        Main function to create window and register callbacks for displaying
        '''
	# Enable texture
	glClearColor(0, 0, 0, 0)
	#glPixelStorei( GL_UNPACK_ALIGNMENT, 1 )
    	#glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_REPEAT ) 
    	#glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_REPEAT )
    	#glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR )
    	#glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR )
    	#glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tw, th, 0, GL_RGB, GL_UNSIGNED_BYTE, Idata)

        # Initialize glut
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(0, 0)
        glutCreateWindow("Simulation")
        #glutMouseFunc(self.mouse_button)
        #glutMotionFunc(self.mouse_motion)
        glutDisplayFunc(self.draw)
        glutIdleFunc(self.draw)
        glutReshapeFunc(self.reshape_func)
        glutKeyboardFunc(self.keyboardFunc)
        glutSpecialFunc(self.specialFunc)

        # Initialize opengl environment
        #glClearColor(0., 0., 0., 1.)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_TEXTURE_2D)
        glShadeModel(GL_SMOOTH)

        # Enter main loop - returns on glutLeaveMainLoop
        glutMainLoop()

    def reshape_func( self, w, h ):
        '''
        Callback to change the camera on a window resize
        '''
        if h == 0: h = 1
	if w == 0: w = 1
        glViewport(0, 0, w, h)
#	vp = glGetIntegerv(GL_VIEWPORT)
#	#glRenderMode(GL_SELECT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
#	#gluPickMatrix(150, vp[3]-150, 6, 6, vp)
        gluPerspective(45.0, float(w)/float(h), 0.0001, 1000.0)

    def keyboardFunc(self, key, x, y):
	scale = 22.5
	# Esc or q to exit
        if (key == b'\033'): sys.exit()

        # WASD movement controls
        elif key == b'W': self.camera.rotateUp(scale*self.step_size)
        elif key == b'w': self.camera.rotateUp(self.step_size)
        elif key == b'Q': self.camera.rotateUp(-scale*self.step_size)
        elif key == b'q': self.camera.rotateUp(-self.step_size)

        elif key == b'Z': self.camera.rotateRight(scale*self.step_size)
        elif key == b'z': self.camera.rotateRight(self.step_size)
        elif key == b'A': self.camera.rotateRight(-scale*self.step_size)
        elif key == b'a': self.camera.rotateRight(-self.step_size)

        elif key == b'X': self.camera.rotateForward(scale*self.step_size)
        elif key == b'x': self.camera.rotateForward(self.step_size)
        elif key == b'S': self.camera.rotateForward(-scale*self.step_size)
        elif key == b's': self.camera.rotateForward(-self.step_size)

	lPosition = self.camera.position
	lDirection = self.camera.forward

	glutPostRedisplay()

    def specialFunc( self, key, x, y ):
	stepsize = 90.0 / (width + height)
	if key == GLUT_KEY_LEFT:
	    self.camera.translate(-self.camera.right * stepsize)
	elif key == GLUT_KEY_RIGHT:
	    self.camera.translate(self.camera.right * stepsize)
	elif key == GLUT_KEY_UP:
 	    self.camera.translate(self.camera.forward * stepsize)
	elif key == GLUT_KEY_DOWN:
 	    self.camera.translate(-self.camera.forward * stepsize)

	lPosition = self.camera.position
	lDirection = self.camera.forward

	glutPostRedisplay()

