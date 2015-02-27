import scipy as sp
import numpy as np
import numpy.linalg as LA
from scipy.integrate import ode
import pylab as plt
import matplotlib.image as mpimg
from joblib import Parallel, delayed
import warnings

#coordinate system:  t, r, theta, phi
# black hole at 0, 0, 0
# screen at 100, 0, 0

m = 1
rs = 2*m
b = 10

def sc_metric(t, r, theta, phi):

	metric = np.zeros((4, 4))
	metric[0, 0] = 1
	metric[1, 1] = -1.0
	metric[2, 2] = - (b**2 + r**2)
	metric[3, 3] = -(b**2 + r**2) * (sp.sin(theta))**2
	
	return metric
	
def sc_geodesic(tau, state):

	dstate = np.zeros(np.shape(state))
	shp = np.shape(state)
	t = state[0]
	r = state[1]
	theta = state[2]
	phi = state[3]
	
	dt = state[4]
	dr = state[5]
	dtheta = state[6]
	dphi = state[7]
	
	dstate[0] = dt
	dstate[1] = dr
	dstate[2] = dtheta
	dstate[3] = dphi
	
	#dstate[4] =  (2*m*dr*dt)/(2*m*r - r**2)
	#dstate[5] = (-m*dr**2/(2*m*r-r**2)) - (2*m - r)*dtheta**2 - (2*m-r)*sp.sin(theta)**2 * dphi**2 + (m*(2*m-r)*dt**2)/(r**3)
	#dstate[6] = -2*dr*dtheta/r + sp.cos(theta)*sp.sin(theta)*dphi**2
	#dstate[7] = -(2*(dr + r*(1.0/sp.tan(theta))*dtheta)*dphi)/r
	
	dstate[4] = 0
	dstate[5] = r*(dtheta**2 + sp.sin(theta)**2 * dphi**2)
	dstate[6] = -1.0*((2*r*dr*dtheta)/(b**2 + r**2)) + sp.cos(theta)*sp.sin(theta)*dphi**2
	dstate[7] = -1.0*((2*r*dr*dphi)/(b**2 + r**2)) - 2*dtheta*dphi/sp.tan(theta)
	
	#if r < rs+0.1:
	#	dstate = sp.zeros(shp)
	#if np.abs(dt) > 20 or np.abs(dstate[7]) > 100:
	#	dstate = sp.zeros(shp)
	
	return dstate

def sphere_to_rect(r, theta, phi):

	x = r * sp.sin(theta) * sp.cos(phi)
	y = r * sp.sin(theta) * sp.sin(phi)
	z = r * sp.cos(theta)
	
	return (x, y, z)



# Image Setup

Nx = 240
Nz = 320

phi_cs_list = np.linspace(0, 1, Nx);
theta_cs_list = np.linspace(0, 1, Nz);

cam_image = sp.zeros([Nx, Nz, 3]);


back_img = np.abs(np.random.randn(1000, 1000))
img = mpimg.imread('neuro2.png')
back_img = img
screen_x_size, screen_z_size, dims = np.shape(back_img)

other_img = mpimg.imread('sunset.png')
other_x_size, other_z_size, dims = np.shape(other_img)

other_img = other_img[:, :, :3]
#Aperture
ap_v = np.pi/6
ap_h = np.pi/6

ri = ode(sc_geodesic).set_integrator('lsoda')


def ray_trace(x, y, cam_pos):
	
	#pix_fido_x, pix_fido_y, pix_fido_z = sphere_to_rect(1, pix_theta, pix_phi)
			
	#print("Pixel: " + str(xind/(Nx*1.0))) 
		#print(pix_fido_x)
		#print(pix_fido_y)
		#print(pix_fido_z)
	
		#ray3 = np.array([pix_fido_x, pix_fido_y, pix_fido_z]);
		
	a = phi_cs_list[x]
	b = theta_cs_list[y]
	ray3 = np.array([-1, -(2*b-1)*sp.tan(ap_v/2) + sp.pi/48, -(2*a -1)*sp.tan(ap_h/2)]);
	
	# convert ray into null vector to get initial conditions

	ray_metric = sc_metric(cam_pos[0], cam_pos[1], cam_pos[2], cam_pos[3])
	ray_3metric = ray_metric[1:, 1:]		
	ray_dir_normsq = np.dot(ray3, np.dot(ray_3metric, ray3))


	ray_t = sp.sqrt(-1.0 * (ray_dir_normsq / ray_metric[0, 0]))
	
	# our initial condition:
	rayv = np.array([ray_t, ray3[0], ray3[1], ray3[2]])
	rayp = cam_pos
	raypinit = rayp
	rayvinit = rayv
		
	initcond = sp.concatenate((rayp, rayv))
	ri.set_initial_value(initcond, 0)
	# propagate until we hit the back wall or until time runs out, whichever happens first
	iter = 0
	dt = 0.5
	#r = camy
	xyzstore = sp.zeros((5000, 3))
	while(ri.successful() and np.abs(ri.y[1]) <= 100):
		iter  = iter+1
		ri.integrate(ri.t+dt)
	soln = ri.y
	r = soln[1]
	theta = soln[2]
	phi = soln[3]
	
	if(np.abs(r) >= 100):
		#print("HitScreen!: " + str(theta) + ", " + str(phi))
		phi_t = sp.mod(np.abs(phi), np.pi)
		theta_t = sp.mod(theta, np.pi)
		#phi = phi - np.pi/2
		
		if phi < 0:
			phi_t = sp.pi - sp.mod(np.abs(phi), np.pi)
		
		screen_prop_x = phi_t/(np.pi)
		screen_prop_z = theta_t/(np.pi)
		#print(str(screen_prop_x) + ",  " + str(screen_prop_z))
		#cam_image[xind, zind] = np.sin(2*np.pi*4*phi)
		imgx = (np.sin(phi) + 1)/2.0
		imgy = (np.cos(theta) + 1)/2.0
		if r > 0:
			screen_pix_x = np.floor(screen_prop_x * screen_x_size)
			screen_pix_z = np.floor(screen_prop_z * screen_z_size)
			return back_img[screen_pix_x, screen_pix_z, :]
		else:
			screen_pix_x = np.floor(screen_prop_x * other_x_size)
			screen_pix_z = np.floor(screen_prop_z * other_z_size)
			return other_img[screen_pix_x, screen_pix_z, :]
			

if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	pts = [(x, y) for x in range(Nx) for y in range(Nz)]
	cam_y_list = np.linspace(25, -20, 60) 
	for ind, camy in enumerate(cam_y_list):
		#camy = 5.0
		print(ind)
		cam = np.array([0, camy, sp.pi/2, -np.pi/2 + ind*sp.pi/120])

		res = Parallel(n_jobs=-1, verbose = 50)(delayed(ray_trace)(x, y, cam) for x, y in pts)

		if sp.size(res)>Nx*Nz:
			cam_image =np.reshape(res, (Nx, Nz, 3))
		else:

			res = np.reshape(res, (Nx, Nz))
			for x, y in pts:
				cam_image[x, y, :] = res[x, y]
		#cam_image = np.reshape(res, (Nx, Nz, 3))
		np.save('wh_out' +str(ind), cam_image)
		
	warnings.resetwarnings()
	
	

		
#imgplt = plt.imshow(cam_image)
#imgplt.set_cmap('gray')
#plt.show()
	
	
	
	
