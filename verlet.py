import numpy as np

def velocity_verlet(t0, t1, dt, force, m, x_0, v_0):
    """
    Velocity Verlet integrator as described in Gromacs manual
    t0 - initial time
    t1 - end time
    dt - time step of integration
    force_field: function f(x), which should return vector of the same shape as x,
    containig force acting on each particle
    m - masses of particles
    x_0 - initial location of particles
    v_0 - initial velocity of particles
    """
    # Number of timesteps
    N = int((t1 - t0) / dt)
    
    # Allocate memory for solution
    X = np.zeros((*x_0.shape, N))
    V = np.zeros((*v_0.shape, N))
    a = np.zeros((*v_0.shape, N))
    
    # Initialize
    X[:, :, 0] = x_0
    V[:, :, 0] = v_0
    a[:, :, 0] = force(x_0) / m[:, None]
    T = np.array([dt * i for i in range(N)])
    
    # Main loop
    for t in range(1, N):
        X[:, :, t] = X[:, :, t - 1] + V[:, :, t - 1] * dt + 0.5 * a[:, :, t - 1] * (dt ** 2)
        a[:, :, t] = force(X[:, :, t]) / m[:, None]
        V[:, :, t] = V[:, :, t - 1] + 0.5 * (a[:, :, t - 1] + a[:, :, t]) * dt
    
    return T, X
