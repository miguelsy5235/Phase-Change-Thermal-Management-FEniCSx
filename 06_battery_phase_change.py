import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import fem
from dolfinx.io import gmsh, VTXWriter
from dolfinx.fem.petsc import LinearProblem
from basix.ufl import element

comm = MPI.COMM_WORLD
rank = comm.rank
TAG = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)

# --------------------------------------------------
# Mesh
# --------------------------------------------------
ret = gmsh.read_from_msh("battery_pcm_quad.msh", comm, rank=0, gdim=2)
domain = ret[0]

x = domain.geometry.x
xmin, ymin = float(np.min(x[:, 0])), float(np.min(x[:, 1]))
xmax, ymax = float(np.max(x[:, 0])), float(np.max(x[:, 1]))

TAG("Quadrilateral Gmsh mesh imported")
TAG(f"Domain size = {xmax-xmin:.4f} x {ymax-ymin:.4f} m")

dx = ufl.Measure("dx", domain=domain)
tdim = domain.topology.dim

# --------------------------------------------------
# Geometry parameters
# --------------------------------------------------
L = 0.10
bx = 0.03
by = 0.02
cx = L / 2
cy = L / 2

batt_xmin = cx - bx / 2
batt_xmax = cx + bx / 2
batt_ymin = cy - by / 2
batt_ymax = cy + by / 2

tol = 1e-10

def inside_battery(x):
    return ( (x[0] >= batt_xmin - tol)
           & (x[0] <= batt_xmax + tol)
           & (x[1] >= batt_ymin - tol)
           & (x[1] <= batt_ymax + tol) )

# --------------------------------------------------
# Water / ice properties
# --------------------------------------------------
rho_pcm = 997.0
cp_pcm = 4182.0
k_pcm = 0.6

rho_batt = 2500.0
cp_batt = 900.0
k_batt = 5.0

Q_batt = 5.0e6

Tm = 273.15
Lh = 150000.0
delta_T = 0.5

# --------------------------------------------------
# Time
# --------------------------------------------------
dt = 0.05
t_end = 300.0
nsteps = int(round(t_end / dt))
save_every = 20     # saves every 1 s

# --------------------------------------------------
# Function spaces
# --------------------------------------------------
cell = domain.ufl_cell().cellname()

Te = element("Lagrange", cell, 1)
Tspace = fem.functionspace(domain, Te)

Vvec = fem.functionspace(domain, element("Lagrange", cell, 1, shape=(tdim,)))

T = ufl.TrialFunction(Tspace)
s = ufl.TestFunction(Tspace)

Tn = fem.Function(Tspace, name="Temperature")
Tn.interpolate(lambda x: np.full(x.shape[1], 268.15))
Tn.x.scatter_forward()

T_viz = fem.Function(Tspace, name="Temperature")
fl_viz = fem.Function(Tspace, name="LiquidFraction")
battery_viz = fem.Function(Tspace, name="BatteryRegion")
u_zero_viz = fem.Function(Vvec, name="Velocity")

u_zero_viz.x.array[:] = 0.0
u_zero_viz.x.scatter_forward()

# --------------------------------------------------
# Region markers
# --------------------------------------------------
battery_marker = fem.Function(Tspace, name="BatteryMarker")
battery_marker.interpolate(lambda x: np.where(inside_battery(x), 1.0, 0.0))
battery_marker.x.scatter_forward()

pcm_marker = fem.Function(Tspace, name="PCMMarker")
pcm_marker.x.array[:] = 1.0 - battery_marker.x.array
pcm_marker.x.scatter_forward()

TAG(f"Initial Tmin = {np.min(Tn.x.array):.2f} K")
TAG(f"Initial Tmax = {np.max(Tn.x.array):.2f} K")
TAG(f"Battery marker min/max = {np.min(battery_marker.x.array):.1f}/{np.max(battery_marker.x.array):.1f}")

# --------------------------------------------------
# Solver
# --------------------------------------------------
opts_temp = { "ksp_type": "preonly", "pc_type": "lu",}

# --------------------------------------------------
# Output
# --------------------------------------------------
def update_output():
    T_viz.x.array[:] = Tn.x.array
    T_viz.x.scatter_forward()

    fl_viz.x.array[:] = 0.5 * (
        1.0 + np.tanh((Tn.x.array - Tm) / delta_T)
    )
    fl_viz.x.scatter_forward()

    battery_viz.x.array[:] = battery_marker.x.array
    battery_viz.x.scatter_forward()

# --------------------------------------------------
# Time loop
# --------------------------------------------------
TAG("Starting water/ice PCM simulation on quadrilateral battery mesh")

with VTXWriter( domain.comm, "results_water_ice_quad_pcm.bp",
    [T_viz, fl_viz, battery_viz, u_zero_viz],)
     as vtx:

    update_output()
    vtx.write(0.0)

    for step in range(1, nsteps + 1):

        t = step * dt

        # Liquid fraction
        fl_n = 0.5 * (1.0 + ufl.tanh((Tn - Tm) / delta_T))

        # Apparent heat capacity
        dfl_dT = 0.5 / delta_T * (  1.0 - ufl.tanh((Tn - Tm) / delta_T) ** 2 )
        cp_eff_pcm = cp_pcm + Lh * dfl_dT

        rho_cp_eff = (
            pcm_marker * rho_pcm * cp_eff_pcm  + battery_marker * rho_batt * cp_batt )

        k_eff = ( pcm_marker * k_pcm + battery_marker * k_batt )

        # Energy equation with enthalpy method
        a_T = ( (rho_cp_eff / dt) * T * s * dx + k_eff * ufl.inner(ufl.grad(T), ufl.grad(s)) * dx)

        L_T = ( (rho_cp_eff / dt) * Tn * s * dx + battery_marker * Q_batt * s * dx)

        problem_T = LinearProblem( a_T, L_T, bcs=[],
            petsc_options_prefix=f"water_ice_quad_{step}_",
            petsc_options=opts_temp,  )

        T_out = problem_T.solve()
        T_out.x.scatter_forward()

        if np.any(np.isnan(T_out.x.array)):
            raise RuntimeError(f"NaN detected at t={t:.2f}s")

        Tn.x.array[:] = T_out.x.array
        Tn.x.scatter_forward()

        if step % save_every == 0 or step == nsteps:

            update_output()
            vtx.write(t)

            Tmin = float(np.min(Tn.x.array))
            Tmax = float(np.max(Tn.x.array))
            dT = Tmax - Tmin

            fl_mean = float(np.mean(fl_viz.x.array))
            fl_max = float(np.max(fl_viz.x.array))

            TAG(
                f"t={t:7.2f}s | "
                f"Tmin={Tmin:7.2f} K | "
                f"Tmax={Tmax:7.2f} K | "
                f"ΔT={dT:7.2f} K | "
                f"fl_mean={fl_mean:5.3f} | "
                f"fl_max={fl_max:5.3f}"
            )

TAG("Simulation completed")
TAG("Saved as results_water_ice_quad_pcm.bp")