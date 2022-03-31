import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(ic, ti, p):
	m1, m2, k1, k2, k3, xeq1, xeq2, xeq3, xp = p
	x1, v1, x2, v2 = ic

	print(ti)

	return [v1, A1.subs({M1:m1, K1:k1, K2:k2, Xeq1:xeq1, Xeq2:xeq2, Xeq3:xeq3, Xp:xp, X1:x1, X2:x2}),\
		v2, A2.subs({M2:m2, K2:k2, K3:k3, Xeq1:xeq1, Xeq2:xeq2, Xeq3:xeq3, Xp:xp, X1:x1, X2:x2})]


M1, M2, K1, K2, K3, Xeq1, Xeq2, Xeq3, Xp, t = sp.symbols('M1 M2 K1 K2 K3 Xeq1 Xeq2 Xeq3 Xp t')
X1, X2 = dynamicsymbols('X1 X2')

X1dot = X1.diff(t, 1)
X2dot = X2.diff(t, 1)

T = sp.Rational(1, 2) * (M1 * X1dot**2 + M2 * X2dot**2)
V = sp.Rational(1, 2) * (K1 * (X1 - Xeq1)**2 + K2 * (X2 - X1 - Xeq2)**2 + K3 * (X2 - Xp + Xeq3)**2)

L = T - V

dLdX1 = L.diff(X1, 1)
dLdX1dot = L.diff(X1dot, 1)
ddtdLdX1dot = dLdX1dot.diff(t, 1)
dL1 = ddtdLdX1dot - dLdX1

dLdX2 = L.diff(X2, 1)
dLdX2dot = L.diff(X2dot, 1)
ddtdLdX2dot = dLdX2dot.diff(t, 1)
dL2 = ddtdLdX2dot - dLdX2

X1ddot = X1.diff(t, 2)
X2ddot = X2.diff(t, 2)

sol = sp.solve([dL1,dL2],(X1ddot,X2ddot))

A1 = sp.simplify(sol[X1ddot])
A2 = sp.simplify(sol[X2ddot])

#-------------------------------------------

m1, m2 = [1, 1]
k1, k2, k3 = [25, 25 ,25]
xeq1, xeq2, xeq3 = [4, 4, 4]
x1o, x2o = [2.5, 6.5]
v1o, v2o = [0, 0]
tf = 60
post2 = 12
rad = 0.25
mass = "proportional"
spring = "proportional" 

p = m1, m2, k1, k2, k3, xeq1, xeq2, xeq3, post2
ic = x1o, v1o, x2o, v2o

nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

xv = odeint(integrate, ic, ta, args=(p,))

ke = np.asarray([T.subs({M1:m1, M2:m2, X1dot:xv[i,1], X2dot:xv[i,3]}) for i in range(nframes)])
pe = np.asarray([V.subs({K1:k1, K2:k2, K3:k3, Xeq1:xeq1, Xeq2:xeq2, Xeq3:xeq3, Xp:post2, X1:xv[i,0], X2:xv[i,2]}) for i in range(nframes)])
E = ke + pe

#-----------------------------------------------

post1 = 0
yline = 0
if mass == "proportional":
	m = np.array([m1, m2])
	ra = rad*m/max(m)
xmax = post2 + rad
xmin = post1 - rad
ymax = yline + 2 * max(ra)
ymin = yline - 2 * max(ra)
nl1 = np.ceil((max(xv[:,0])-ra[0])/(2*ra[0]))
nl2 = np.ceil((max(xv[:,2]-xv[:,0])-(ra[0]+ra[1]))/(ra[0]+ra[1]))
nl3 = np.ceil((post2-min(xv[:,2])-ra[1])/(2*ra[1]))
nl = np.array([nl1, nl2, nl3])
if spring == "proportional":
	k = np.array([k1, k2, k3])
	nl = nl*k/min(k)
	nl = np.ceil(nl[:])
xl1 = np.zeros((int(nl[0]),nframes))
yl1 = np.zeros((int(nl[0]),nframes))
xl2 = np.zeros((int(nl[1]),nframes))
yl2 = np.zeros((int(nl[1]),nframes))
xl3 = np.zeros((int(nl[2]),nframes))
yl3 = np.zeros((int(nl[2]),nframes))
for i in range(nframes):
	l1 = (xv[i,0] - post1 - ra[0])/nl[0]
	l2 = (xv[i,2] - xv[i,0] - (ra[0] + ra[1]))/nl[1]
	l3 = (post2 - xv[i,2] - ra[1])/nl[2]
	xl1[0][i] = xv[i,0] - ra[0] - 0.5 * l1
	xl2[0][i] = xv[i,0] + ra[0] + 0.5 * l2
	xl3[0][i] = xv[i,2] + ra[1] + 0.5 * l3
	for j in range(1,int(nl[0])):
		xl1[j][i] = xl1[j-1][i] - l1
	for j in range(int(nl[0])):
		yl1[j][i] = yline+((-1)**j)*(np.sqrt(ra[0]**2 - (0.5*l1)**2))
	for j in range(1,int(nl[1])):
		xl2[j][i] = xl2[j-1][i] + l2
	for j in range(int(nl[1])):
		yl2[j][i] = yline+((-1)**j)*(np.sqrt(((ra[0]+ra[1])/2)**2 - (0.5*l2)**2))
	for j in range(1,int(nl[2])):
		xl3[j][i] = xl3[j-1][i] + l3
	for j in range(int(nl[2])):
		yl3[j][i] = yline+((-1)**j)*(np.sqrt(ra[1]**2 - (0.5*l3)**2))

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((xv[frame,0],yline),radius=ra[0],fc='xkcd:red')
	plt.gca().add_patch(circle)
	circle=plt.Circle((xv[frame,2],yline),radius=ra[1],fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.plot([post1,post1],[ymin,ymax],'xkcd:cerulean',lw=4)
	plt.plot([post2,post2],[ymin,ymax],'xkcd:cerulean',lw=4)
	plt.plot([xv[frame,0]-ra[0],xl1[0][frame]],[yline,yl1[0][frame]],'xkcd:cerulean')
	plt.plot([xl1[int(nl[0])-1][frame],post1],[yl1[int(nl[0])-1][frame],yline],'xkcd:cerulean')
	for i in range(int(nl[0])-1):
		plt.plot([xl1[i][frame],xl1[i+1][frame]],[yl1[i][frame],yl1[i+1][frame]],'xkcd:cerulean')
	plt.plot([xv[frame,0]+ra[0],xl2[0][frame]],[yline,yl2[0][frame]],'xkcd:cerulean')
	plt.plot([xl2[int(nl[1])-1][frame],xv[frame,2]-ra[1]],[yl2[int(nl[1])-1][frame],yline],'xkcd:cerulean')
	for i in range(int(nl[1])-1):
		plt.plot([xl2[i][frame],xl2[i+1][frame]],[yl2[i][frame],yl2[i+1][frame]],'xkcd:cerulean')
	plt.plot([xv[frame,2]+ra[1],xl3[0][frame]],[yline,yl3[0][frame]],'xkcd:cerulean')
	plt.plot([xl3[int(nl[2])-1][frame],post2],[yl3[int(nl[2])-1][frame],yline],'xkcd:cerulean')
	for i in range(int(nl[2])-1):
		plt.plot([xl3[i][frame],xl3[i+1][frame]],[yl3[i][frame],yl3[i+1][frame]],'xkcd:cerulean')
	plt.title("Three Springs")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['V','T','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('three_spring_chain.mp4', writer=writervideo)
plt.show()













