import control
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


M = 1
m = 0.5
L = 0.4
g = 9.81
b = 0.01

detL = ((M + m) * m* L**2) - m**2 * L**2

def nonlinearpen(state, F, M, m, L, b, g):
    # x[0] cart position
    # x[1] cart velocity
    # x[2] pendulum angle
    # x[3] pendulum angular velocity


    x = state

    det = ((M + m) * m* L**2) - m**2 * L**2 * (np.cos(x[2])**2)

    xaccel = (1/det)*(m**2 * L**3 * np.sin(x[2])*x[3]**2 
                      + m*L**2 * F 
                      - m * L**2 * b * x[1] 
                      - m**2 * L**2 * g * np.cos(x[2])* np.sin(x[2]))
    
    thetaaccel = (1/det)*((M + m)*(m*g*L*np.sin(x[2])) 
                          - m**2 * L**2 * np.cos(x[2]) * np.sin(x[2]) * x[3]**2 
                          - (m*L*np.cos(x[2]))
                          *(F - b * x[1] ))

    return [x[1], xaccel, x[3], thetaaccel]

A = np.array([
    [0, 1, 0, 0],
    [0, -(m * L**2 * b)/detL, -(m**2 * L**2 * g)/detL, 0],
    [0, 0, 0, 1],
    [0, (m*L*b)/detL, ((M+m)*(m*g*L))/detL, 0]
])

B = np.array([
    [0],
    [m*L**2/detL],
    [0],
    [-m*L/detL]
])

Q = np.diag([5,1,100,1])
R = np.array([[1]])

K, S, E = control.lqr(A, B, Q, R)

K = K.flatten()


def CLpen(t,state):
    F = float(-K @ state)

    return nonlinearpen(state, F, M, m, L, b, g)

# initang = np.random.uniform(0,10)
# print(initang)
init = [0.0, 0.0, np.radians(5), 0.0]

t = (0,5)

sol = solve_ivp(CLpen,t,init, max_step=0.005, rtol=1e-8)

t     = sol.t
x     = sol.y[0] # cart position
xdot  = sol.y[1] # cart velocity
theta = sol.y[2] # pend angle
thetadot = sol.y[3] # pend velocity

fig, axes = plt.subplots(2,1, figsize=(10,8), sharex=True)

axes[0].plot(t,x, label='X Cart Pos')
axes[0].plot(t,xdot, label='xdot cart velocity', linestyle='--')
axes[0].set_ylabel('Cart')
axes[0].legend()
axes[0].grid(True, alpha = 0.3)

axes[1].plot(t,np.degrees(theta), label='Pendulum Angle')
axes[1].plot(t,np.degrees(thetadot), label='Pendulum angular velocity', linestyle='--')
axes[1].set_ylabel('Pendulum')
axes[1].set_xlabel('Time (s)')
axes[0].set_title('LQR controller for inverted pendulum with initial angle 5 degrees')
axes[1].legend()
axes[1].grid(True, alpha = 0.3)
plt.savefig("Question_1.png")
# plt.suptitle('LQR controller for inverted pendulum')
# plt.show()

framesPerSec = 60
frame_time = np.arange(0, t[-1], 1/framesPerSec)
x_frames = np.interp(frame_time, t, x) #https://numpy.org/devdocs/reference/generated/numpy.interp.html
theta_frames = np.interp(frame_time, t, theta)

fig2, ax = plt.subplots(figsize=(8,6))
cw, ch = 0.3, 0.18
cart = plt.Rectangle((0, -ch), cw, ch, color='blue')
ax.add_patch(cart)
rod, = ax.plot([], [], lw=2, color='red')
bob, = ax.plot([], [], 'o', color='green', ms=10)
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 1)
ax.set_aspect('equal')
ax.set_title('LQR controller for inverted pendulum for initial angle 5 degrees')
def update(frame):
    xc = x_frames[frame]
    thetac = theta_frames[frame]
    cart.set_xy((xc - cw/2, -ch))
    bx = xc+ L * np.sin(thetac) # Forward kinematics to get bob position
    by = L * np.cos(thetac)
    rod.set_data([xc, bx], [0, by]) # updating position of rod
    bob.set_data([bx], [by])
    return cart, rod, bob
# Reference https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
anim = FuncAnimation(fig2, update, frames=len(frame_time), blit=True, interval=1000/framesPerSec)
# Ref https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html
anim.save("Question_1.gif", writer=PillowWriter(fps=framesPerSec))
plt.show()