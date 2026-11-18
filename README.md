# UAV 3D Flight Control Simulation: Cascade PID vs. LQR under Wind Disturbances

Interactive six-degree-of-freedom (6-DOF) three-dimensional flight control simulation framework for a quadcopter Unmanned Aerial Vehicle (UAV), developed in **Python**. This software enables real-time parameter hot-tuning and quantitative dynamic performance comparison between a **Hierarchical Cascade PID Controller** and an **Infinite-Horizon Linear Quadratic Regulator (LQR)** subject to continuous stochastic Dryden turbulence and discrete impulsive wind gusts.

---

## Repository Structure

```text
quadcopter_simulation/
│
├── main.py                  # Main desktop application entry point and OS taskbar integration
├── config.py                # Physical constants, actuator limits, and initial tuning matrices
├── requirements.txt         # Pinned scientific and GUI dependencies
├── README.md                # Technical documentation and mathematical formulation
├── run.bat                  # Windows execution launcher
├── test_flight_control.py   # Automated unit and GUI integration test suite
│
├── assets/
│   ├── drone_icon.ico       # Multi-resolution application icon for window title bar and taskbar
│   └── drone_icon.png       # High-resolution vector quadcopter emblem
│
├── physics/
│   ├── __init__.py
│   └── quadcopter_model.py  # 6-DOF Newton-Euler dynamics, RK4 solver, and wind disturbance model
│
├── controllers/
│   ├── __init__.py
│   ├── pid_controller.py    # Cascade PID controller (Position -> Attitude -> Rotor Allocation)
│   └── lqr_controller.py    # State-space linearization and Continuous Algebraic Riccati solver
│
└── ui/
    ├── __init__.py
    └── main_window.py       # PySide6 split-pane GUI, 60-FPS 3D viewport, and PyQtGraph telemetry
```

---

## Mathematical Modeling and Theoretical Framework (6-DOF)

### 1. State Vector and Coordinate Reference Frames
The quadcopter is modeled as a rigid body evolving in three-dimensional space with respect to an inertial reference frame $\mathcal{I}$ adopting the **East-North-Up (ENU)** convention ($X$ East, $Y$ North, $Z$ Vertical Altitude) and a body-fixed frame $\mathcal{B}$ centered at the vehicle Center of Gravity (CoG).

The twelve-dimensional state vector $\mathbf{x} \in \mathbb{R}^{12}$ is defined as:

$$\mathbf{x} = \begin{bmatrix} x & y & z & v_x & v_y & v_z & \phi & \theta & \psi & p & q & r \end{bmatrix}^T$$

- **Inertial Position:** $\mathbf{p} = [x, y, z]^T \in \mathbb{R}^3\ [\text{m}]$
- **Inertial Linear Velocity:** $\mathbf{v} = [v_x, v_y, v_z]^T = \dot{\mathbf{p}} \in \mathbb{R}^3\ [\text{m/s}]$
- **Euler Attitude Angles (Intrinsic Z-Y-X Convention):** $\boldsymbol{\eta} = [\phi, \theta, \psi]^T \in \mathbb{R}^3\ [\text{rad}]$ — Roll ($\phi$), Pitch ($\theta$), and Yaw ($\psi$).
- **Body-Frame Angular Velocity:** $\boldsymbol{\omega} = [p, q, r]^T \in \mathbb{R}^3\ [\text{rad/s}]$

### 2. Attitude Kinematics and Rotation Matrix
The transformation from the body-fixed frame $\mathcal{B}$ to the inertial frame $\mathcal{I}$ is parameterized by the orthogonal direction cosine matrix $\mathbf{R}(\phi, \theta, \psi) = \mathbf{R}_z(\psi)\mathbf{R}_y(\theta)\mathbf{R}_x(\phi) \in SO(3)$:

$$\mathbf{R} = \begin{bmatrix} \cos\theta\cos\psi & \sin\phi\sin\theta\cos\psi - \cos\phi\sin\psi & \cos\phi\sin\theta\cos\psi + \sin\phi\sin\psi \\ \cos\theta\sin\psi & \sin\phi\sin\theta\sin\psi + \cos\phi\cos\psi & \cos\phi\sin\theta\sin\psi - \sin\phi\cos\psi \\ -\sin\theta & \sin\phi\cos\theta & \cos\phi\cos\theta \end{bmatrix}$$

The kinematic differential equation relating body angular rates $\boldsymbol{\omega} = [p, q, r]^T$ to the Euler angle derivatives $\dot{\boldsymbol{\eta}} = [\dot{\phi}, \dot{\theta}, \dot{\psi}]^T$ is given by:

$$\begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix} = \mathbf{W}(\phi, \theta)\begin{bmatrix} p \\ q \\ r \end{bmatrix} = \begin{bmatrix} 1 & \sin\phi\tan\theta & \cos\phi\tan\theta \\ 0 & \cos\phi & -\sin\phi \\ 0 & \frac{\sin\phi}{\cos\theta} & \frac{\cos\phi}{\cos\theta} \end{bmatrix} \begin{bmatrix} p \\ q \\ r \end{bmatrix}$$

### 3. Translational and Rotational Dynamics (Newton-Euler Equations)
Applying Newton's Second Law in the inertial frame $\mathcal{I}$ and Euler's rotational equations of motion in the body-fixed frame $\mathcal{B}$ yields the nonlinear system:

$$m \dot{\mathbf{v}} = \mathbf{R}(\phi, \theta, \psi) \begin{bmatrix} 0 \\ 0 \\ U_1 \end{bmatrix} - \begin{bmatrix} 0 \\ 0 \\ m g \end{bmatrix} + \mathbf{F}_d(\mathbf{v}_{rel})$$

$$\mathbf{J}\dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{J}\boldsymbol{\omega}) + \boldsymbol{\tau}_{gyro} = \begin{bmatrix} \tau_\phi \\ \tau_\theta \\ \tau_\psi \end{bmatrix} + \boldsymbol{\tau}_{wind} - \mathbf{D}_\omega \boldsymbol{\omega}$$

where:
- $\mathbf{J} = \text{diag}(I_{xx}, I_{yy}, I_{zz})$ denotes the principal inertia tensor of the airframe.
- $\boldsymbol{\tau}_{gyro} = J_r (\boldsymbol{\omega} \times \mathbf{e}_3) \Omega_r$ captures the **gyroscopic precession torque** induced by the four counter-rotating rotors with net residual angular speed $\Omega_r = \omega_1 - \omega_2 + \omega_3 - \omega_4$.
- Time integration is performed deterministically via the explicit **4th-Order Runge-Kutta (RK4)** solver at a fixed step size $\Delta t = 10\text{ ms}$ ($100\text{ Hz}$).

### 4. Control Allocation and Rotor Mixing Matrix
For a symmetric **X-configuration** quadcopter with arm length $L$ and moment arm projection $l = L/\sqrt{2}$, the control allocation matrix $\mathbf{M}_{alloc}$ maps the individual rotor thrusts $\mathbf{F} = [F_1, F_2, F_3, F_4]^T$ ($F_i = K_F \omega_i^2$) to the generalized control wrench $\mathbf{u} = [U_1, \tau_\phi, \tau_\theta, \tau_\psi]^T$:

$$\begin{bmatrix} U_1 \\ \tau_\phi \\ \tau_\theta \\ \tau_\psi \end{bmatrix} = \begin{bmatrix} 1 & 1 & 1 & 1 \\ l & -l & -l & l \\ -l & -l & l & l \\ c_M & -c_M & c_M & -c_M \end{bmatrix} \begin{bmatrix} F_1 \\ F_2 \\ F_3 \\ F_4 \end{bmatrix}, \qquad c_M = \frac{K_M}{K_F}$$

### 5. Atmospheric Wind Disturbance and Aerodynamic Drag Model
The `WindDisturbance` module superimposes steady ambient flow, continuous stochastic turbulence shaped by a low-altitude **Dryden spectral filter**, and user-triggered discrete gusts along the inertial $X$, $Y$, and $Z$ axes:

$$\mathbf{v}_{rel} = \mathbf{v} - \mathbf{v}_{wind}(t), \qquad \mathbf{F}_d = -\frac{1}{2} \rho C_d \mathbf{A}_{eff} \mathbf{v}_{rel} |\mathbf{v}_{rel}| - \mathbf{C}_{flap}\mathbf{v}_{rel}$$

Overturning moments $\boldsymbol{\tau}_{wind} = \mathbf{r}_{cp} \times (\mathbf{R}^T \mathbf{F}_d) + \boldsymbol{\tau}_{inflow}$ arise from the vertical offset $\mathbf{r}_{cp}$ of the rotor thrust plane above the center of mass and translational inflow asymmetry.

---

## Flight Control Architectures

### 1. Hierarchical Cascade PID Controller (`controllers/pid_controller.py`)
- **Outer Translational Loop (3D Position & Linear Velocity):** Evaluates a PID control law with integral anti-windup clamping over the position tracking error $\mathbf{e}_p = \mathbf{p}_{ref} - \mathbf{p}$ to compute the desired acceleration vector $\mathbf{a}_{des}$. Applies gravitational feedforward and non-linear tilt compensation $1/(\cos\phi\cos\theta)$ for the collective thrust $U_1$, while projecting horizontal accelerations into the heading-aligned frame to generate attitude angle references:
  $$\theta_{ref} = \arctan2(a_{x,\psi}, g), \qquad \phi_{ref} = \arctan2(-a_{y,\psi}, g)$$
- **Inner Attitude Loop (Euler Angles & Gyroscope Rates):** High-bandwidth inner loop computing the stabilizing control moments $[\tau_\phi, \tau_\theta, \tau_\psi]^T$ scaled by the inertia tensor $\mathbf{J}$ from attitude tracking errors and direct angular rate damping on $[p, q, r]^T$.

### 2. Linear Quadratic Regulator — LQR (`controllers/lqr_controller.py`)
Linearizing the 6-DOF nonlinear dynamics via analytical Jacobian evaluation around the hover equilibrium point ($\mathbf{x}_{eq}, \mathbf{u}_{eq} = [mg, 0, 0, 0]^T$):

$$\delta\dot{\mathbf{x}} = \mathbf{A}\delta\mathbf{x} + \mathbf{B}\delta\mathbf{u}, \qquad \mathbf{A} \in \mathbb{R}^{12\times 12},\ \mathbf{B} \in \mathbb{R}^{12\times 4}$$

The infinite-horizon quadratic cost functional:

$$J = \int_0^{\infty} \left( \delta\mathbf{x}^T \mathbf{Q} \delta\mathbf{x} + \delta\mathbf{u}^T \mathbf{R} \delta\mathbf{u} \right) dt$$

is minimized by solving the **Continuous-Time Algebraic Riccati Equation (CARE)** using `scipy.linalg.solve_continuous_are`:

$$\mathbf{A}^T\mathbf{P} + \mathbf{P}\mathbf{A} - \mathbf{P}\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^T\mathbf{P} + \mathbf{Q} = \mathbf{0} \implies \mathbf{K} = \mathbf{R}^{-1}\mathbf{B}^T\mathbf{P}$$

Whenever the diagonal entries of $\mathbf{Q}$ or $\mathbf{R}$ are adjusted via the interface sliders, the Riccati equation is solved in real time to update the full-state feedback gain matrix $\mathbf{K} \in \mathbb{R}^{4\times 12}$ under the control law $\mathbf{u} = \mathbf{u}_{eq} - \mathbf{K}(\mathbf{x} - \mathbf{x}_{ref})$.

---

## Installation and Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Desktop Simulation Environment
```bash
python main.py
```
*(On Windows systems, the simulator can also be launched via `.\run.bat`).*

### 3. Run the Automated Verification Test Suite
```bash
python test_flight_control.py
```

---

## User Interface and Workstation Controls

1. **Full-Window Interactive 3D Viewport (100% Central Workspace):**
   - On launch, the 3D viewport occupies 100% of the main window with all side drawers collapsed at the edges, providing an unobstructed view of the 3D flight arena.
   - **Left-Click + Drag:** Orbit 3D camera (Azimuth and Elevation).
   - **Right-Click + Drag:** Pan 3D camera focal target.
   - **Scroll Wheel:** Zoom in or out.
   - **`Separate UAVs` Toggle:** In *Live Comparison* mode, visually offsets the PID and LQR quadcopters by $\pm 0.85\text{ m}$ along the $Y$ axis to facilitate side-by-side structural and attitude inspection.
2. **Animated Collapsible Right Side Drawer (`Control Settings`):**
   - Starts collapsed by default. Clicking the `< Control Settings` edge tab (or top-bar toggle) smoothly slides out the control drawer via `QPropertyAnimation` (`QEasingCurve.InOutQuad`).
   - Switch dynamically between **`PID`**, **`LQR`**, and **`Live Comparison (PID vs. LQR)`**.
   - Inject discrete step/impulse gusts along the **Lateral ($X$)**, **Frontal ($Y$)**, or **Vertical ($Z$)** axes, or enable **Continuous Stochastic Wind (Dryden)**.
   - Adjust PID loop gains or LQR weighting matrices $\mathbf{Q}$ and $\mathbf{R}$ on the fly with detailed didactic tooltips explaining each parameter's physical role.
3. **Animated Collapsible Left Side Drawer (`Live Telemetry`):**
   - Starts collapsed by default. Clicking the `Live Telemetry >` edge tab smoothly expands four high-frequency `pyqtgraph` plots: (1) 3D Position Error relative to Setpoint, (2) Euler Attitude Angles ($\phi, \theta$) during gust rejection, (3) Actuator Control Effort / Rotor Motor Power (% PWM), and (4) Applied Wind Velocity Profile.
