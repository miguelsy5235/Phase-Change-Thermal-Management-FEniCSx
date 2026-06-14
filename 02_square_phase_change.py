import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from dolfinx import mesh as dmesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter
from basix.ufl import element, mixed_element

comm = MPI.COMM_WORLD
rank = comm.rank
TAG = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)

# Mesh
L_target = 0.02
n = 40

mesh = dmesh.create_rectangle(
    comm,
    points=((0.0, 0.0), (L_target, L_target)),
    n=(n, n),
    cell_type=dmesh.CellType.quadrilateral,
)

x = mesh.geometry.x
xmin, ymin = float(np.min(x[:, 0])), float(np.min(x[:, 1]))
xmax, ymax = float(np.max(x[:, 0])), float(np.max(x[:, 1]))

dx = ufl.Measure("dx", domain=mesh)
dim = mesh.topology.dim
facet_dim = dim - 1
tol = 1e-12

TAG(f"Simple mesh created: {xmax-xmin:.6f} x {ymax-ymin:.6f} m")

def on_bottom(p):
    return np.isclose(p[1], ymin, atol=tol)

def on_top(p):
    return np.isclose(p[1], ymax, atol=tol)

def on_all_walls(p):
    return (
        np.isclose(p[0], xmin, atol=tol)
        | np.isclose(p[0], xmax, atol=tol)
        | np.isclose(p[1], ymin, atol=tol)
        | np.isclose(p[1], ymax, atol=tol)
    )

# Properties
rho = 997.0
mu = 1.002e-3
nu = mu / rho
k = 0.6
cp = 4182.0

beta = 2.07e-4
g = 9.81
g_vec = ufl.as_vector((0.0, -g))

Tm = 273.15
Lh = 80000.0
delta_T = 1.0

C_mushy = 50.0
eps = 1e-3

# Temperatures
T_bottom = 290.15
T_top = 268.15
T_init = 268.15
T_ref = Tm

# Time
dt = 0.02
t_end = 200.0
nsteps = int(round(t_end / dt))
save_every = 50

# Spaces
cell = mesh.ufl_cell().cellname()

Ve = element("Lagrange", cell, 2, shape=(dim,))
Qe = element("Lagrange", cell, 1)
Te = element("Lagrange", cell, 1)

W = fem.functionspace(mesh, mixed_element([Ve, Qe]))
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()
Tspace = fem.functionspace(mesh, Te)

V1 = fem.functionspace(mesh, element("Lagrange", cell, 1, shape=(dim,)))
Q1 = fem.functionspace(mesh, element("Lagrange", cell, 1))

# Unknowns
U, P = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)

T = ufl.TrialFunction(Tspace)
s = ufl.TestFunction(Tspace)

# Previous fields
u_n = fem.Function(V)
u_n.x.array[:] = 0.0

p_n = fem.Function(Q)
p_n.x.array[:] = 0.0

Tn = fem.Function(Tspace)
Tn.interpolate(lambda x: np.full(x.shape[1], T_init))
Tn.x.scatter_forward()

# Output fields
u_viz = fem.Function(V1, name="Velocity")
p_viz = fem.Function(Q1, name="Pressure")
T_viz = fem.Function(Tspace, name="Temperature")
fl_viz = fem.Function(Tspace, name="LiquidFraction")

# Velocity BC
facets_all = dmesh.locate_entities_boundary(mesh, facet_dim, on_all_walls)

u_zero = fem.Function(V)
u_zero.x.array[:] = 0.0

dofs_u = fem.locate_dofs_topological((W.sub(0), V), facet_dim, facets_all)
bc_u = fem.dirichletbc(u_zero, dofs_u, W.sub(0))

# Pressure gauge
def at_corner(p):
    return np.isclose(p[0], xmin, atol=tol) & np.isclose(p[1], ymin, atol=tol)

p_zero = fem.Function(Q)
p_zero.x.array[:] = 0.0

dofs_p0 = fem.locate_dofs_geometrical((W.sub(1), Q), at_corner)
bc_p0 = fem.dirichletbc(p_zero, dofs_p0, W.sub(1))

# Temperature BC
facets_bottom = dmesh.locate_entities_boundary(mesh, facet_dim, on_bottom)
facets_top = dmesh.locate_entities_boundary(mesh, facet_dim, on_top)

dofs_bottom = fem.locate_dofs_topological(Tspace, facet_dim, facets_bottom)
dofs_top = fem.locate_dofs_topological(Tspace, facet_dim, facets_top)

bc_bottom = fem.dirichletbc(PETSc.ScalarType(T_bottom), dofs_bottom, Tspace)
bc_top = fem.dirichletbc(PETSc.ScalarType(T_top), dofs_top, Tspace)

TAG(f"Velocity wall dofs: {len(dofs_u[0]) if isinstance(dofs_u, tuple) else len(dofs_u)}")
TAG(f"Bottom temperature dofs: {len(dofs_bottom)}")
TAG(f"Top temperature dofs: {len(dofs_top)}")

# Solver options
opts_stokes = {
    "ksp_type": "gmres",
    "pc_type": "lu",
    "ksp_rtol": 1e-8,
}

opts_temp = {
    "ksp_type": "preonly",
    "pc_type": "lu",
}

gamma_div = PETSc.ScalarType(1e-8)
eta_brink = PETSc.ScalarType(1e-10)

def update_output():
    u_viz.interpolate(u_n)
    p_viz.interpolate(p_n)

    T_viz.x.array[:] = Tn.x.array
    T_viz.x.scatter_forward()

    fl_viz.x.array[:] = 0.5 * (
        1.0 + np.tanh((Tn.x.array - Tm) / delta_T)
    )
    fl_viz.x.scatter_forward()

    u_viz.x.scatter_forward()
    p_viz.x.scatter_forward()

TAG("Starting phase-change simulation with velocity")

with VTXWriter(
    mesh.comm,
    "results_phase_change_fast_velocity.bp",
    [u_viz, p_viz, T_viz, fl_viz],
) as vtx:

    update_output()
    vtx.write(0.0)

    for step in range(1, nsteps + 1):

        t = step * dt

        fl_array = 0.5 * (1.0 + np.tanh((Tn.x.array - Tm) / delta_T))
        fl_max = float(np.max(fl_array))
        fl_mean = float(np.mean(fl_array))

        # Activate velocity only when enough liquid exists
        solve_flow = (fl_max > 0.7) and (fl_mean > 0.03)

        fl_n = 0.5 * (1.0 + ufl.tanh((Tn - Tm) / delta_T))

        dfl_dT = 0.5 / delta_T * (
            1.0 - ufl.tanh((Tn - Tm) / delta_T) ** 2
        )

        cp_eff = cp + Lh * dfl_dT
        mushy = C_mushy * ((1.0 - fl_n) ** 2) / (fl_n ** 3 + eps)

        # --------------------------------------------------
        # Velocity / pressure solve
        # --------------------------------------------------
        if solve_flow:

            ramp = min(1.0, max(0.0, (fl_mean - 0.03) / 0.05))

            a_mom = (
                (1.0 / dt) * ufl.inner(U, v) * dx
                + 2.0 * nu * ufl.inner(
                    ufl.sym(ufl.grad(U)),
                    ufl.sym(ufl.grad(v))
                ) * dx
                - ufl.inner(P, ufl.div(v)) * dx
                + ufl.inner(ufl.div(U), q) * dx
                + gamma_div * ufl.inner(ufl.div(U), ufl.div(v)) * dx
                + eta_brink * ufl.inner(U, v) * dx
                + mushy * ufl.inner(U, v) * dx
            )

            L_mom = (
                (1.0 / dt) * ufl.inner(u_n, v) * dx
                - ramp * fl_n * beta
                * ufl.inner((Tn - T_ref) * g_vec, v) * dx
            )

            problem_NS = LinearProblem(
                a_mom,
                L_mom,
                bcs=[bc_u, bc_p0],
                petsc_options_prefix=f"ns_fast_phase_{step}_",
                petsc_options=opts_stokes,
            )

            w_sol = problem_NS.solve()

            Uh, map_u = W.sub(0).collapse()
            Ph, map_p = W.sub(1).collapse()

            u_out = fem.Function(Uh)
            u_out.x.array[:] = w_sol.x.array[map_u]

            p_out = fem.Function(Ph)
            p_out.x.array[:] = w_sol.x.array[map_p]

            if np.any(np.isnan(u_out.x.array)):
                u_n.x.array[:] = 0.0
                p_n.x.array[:] = 0.0
                solve_flow = False
            else:
                u_n.interpolate(u_out)
                p_n.interpolate(p_out)

        else:
            u_n.x.array[:] = 0.0
            p_n.x.array[:] = 0.0

        # --------------------------------------------------
        # Energy solve with enthalpy + convection
        # --------------------------------------------------
        a_T = (
            (rho * cp_eff / dt) * T * s * dx
            + k * ufl.inner(ufl.grad(T), ufl.grad(s)) * dx
            + rho * cp * fl_n * ufl.inner(u_n, ufl.grad(T)) * s * dx
        )

        L_T = (rho * cp_eff / dt) * Tn * s * dx

        problem_T = LinearProblem(
            a_T,
            L_T,
            bcs=[bc_bottom, bc_top],
            petsc_options_prefix=f"temp_fast_phase_{step}_",
            petsc_options=opts_temp,
        )

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
            fl_mean_out = float(np.mean(fl_viz.x.array))
            fl_max_out = float(np.max(fl_viz.x.array))

            u_arr = u_viz.x.array.reshape((-1, dim))
            umax = float(np.max(np.linalg.norm(u_arr, axis=1)))

            TAG(
                f"t={t:7.2f}s | "
                f"Tmin={Tmin:7.2f} K | "
                f"Tmax={Tmax:7.2f} K | "
                f"fl_mean={fl_mean_out:5.3f} | "
                f"fl_max={fl_max_out:5.3f} | "
                f"flow={'on ' if solve_flow else 'off'} | "
                f"umax={umax:.3e} m/s"
            )

TAG("Simulation completed")
TAG("Saved as results_phase_change_fast_velocity.bp") 