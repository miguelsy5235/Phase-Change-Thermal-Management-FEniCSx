# -*- coding: utf-8 -*-

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

# --------------------------------------------------
# SIMPLE MESH, NO GMSH
# --------------------------------------------------
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
H = max(xmax - xmin, ymax - ymin)

TAG(f"TAG: simple mesh created: {xmax-xmin:.6f} x {ymax-ymin:.6f} m")

dx = ufl.Measure("dx", domain=mesh)
dim = mesh.topology.dim
facet_dim = dim - 1
tol = 1e-12

def on_bottom(p):
    return np.isclose(p[1], ymin, atol=tol)

def on_top(p):
    return np.isclose(p[1], ymax, atol=tol)

# --------------------------------------------------
# WATER PROPERTIES
# --------------------------------------------------
rho0 = 997.0
mu = 1.002e-3
nu = mu / rho0

k = 0.6
cp = 4182.0
alpha = k / (rho0 * cp)

beta = 2.07e-4
g = 9.81
g_vec = ufl.as_vector((0.0, -g))

# --------------------------------------------------
# TIME
# --------------------------------------------------
t_end = 100.0
dt = 0.02
nsteps = int(round(t_end / dt))
save_every = 50

# --------------------------------------------------
# FUNCTION SPACES
# --------------------------------------------------
cell = mesh.ufl_cell().cellname()

Ve = element("Lagrange", cell, 2, shape=(dim,))
Qe = element("Lagrange", cell, 1)
Te = element("Lagrange", cell, 2)

W = fem.functionspace(mesh, mixed_element([Ve, Qe]))
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()
Th = fem.functionspace(mesh, Te)

# Visualization spaces
V1 = fem.functionspace(mesh, element("Lagrange", cell, 1, shape=(dim,)))
Q1 = fem.functionspace(mesh, element("Lagrange", cell, 1))
T1 = fem.functionspace(mesh, element("Lagrange", cell, 1))

u_viz = fem.Function(V1, name="Velocity")
p_viz = fem.Function(Q1, name="Pressure")
T_viz = fem.Function(T1, name="Temperature")

# --------------------------------------------------
# VELOCITY BC: NO-SLIP ALL WALLS
# --------------------------------------------------
u_zero = fem.Function(V)
u_zero.x.array[:] = 0.0

facets_all = dmesh.locate_entities_boundary(
    mesh,
    facet_dim,
    lambda p: np.isclose(p[0], xmin, atol=tol)
    | np.isclose(p[0], xmax, atol=tol)
    | np.isclose(p[1], ymin, atol=tol)
    | np.isclose(p[1], ymax, atol=tol),
)

dofs_u = fem.locate_dofs_topological((W.sub(0), V), facet_dim, facets_all)
bc_u = fem.dirichletbc(u_zero, dofs_u, W.sub(0))

# --------------------------------------------------
# TEMPERATURE BC
# Bottom hot, top cold, lateral walls adiabatic naturally
# --------------------------------------------------
T_bottom = 298.0
T_top = 293.0
T_init = 293.0
T_ref = 293.0

facets_bottom = dmesh.locate_entities_boundary(mesh, facet_dim, on_bottom)
facets_top = dmesh.locate_entities_boundary(mesh, facet_dim, on_top)

dofs_Tb = fem.locate_dofs_topological(Th, facet_dim, facets_bottom)
dofs_Ttop = fem.locate_dofs_topological(Th, facet_dim, facets_top)

TAG(f"Bottom temperature dofs: {len(dofs_Tb)}")
TAG(f"Top temperature dofs: {len(dofs_Ttop)}")

bc_T_bottom = fem.dirichletbc(PETSc.ScalarType(T_bottom), dofs_Tb, Th)
bc_T_top = fem.dirichletbc(PETSc.ScalarType(T_top), dofs_Ttop, Th)

# --------------------------------------------------
# PRESSURE GAUGE: p = 0 at bottom-left corner
# --------------------------------------------------
def at_corner(p):
    return np.isclose(p[0], xmin, atol=tol) & np.isclose(p[1], ymin, atol=tol)

p_pin = fem.Function(Q)
p_pin.x.array[:] = 0.0

dofs_p0 = fem.locate_dofs_geometrical((W.sub(1), Q), at_corner)
bc_p0 = fem.dirichletbc(p_pin, dofs_p0, W.sub(1))

# --------------------------------------------------
# VARIABLES AND INITIAL CONDITIONS
# --------------------------------------------------
U, P = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)

u_n = fem.Function(V)
u_n.x.array[:] = 0.0

p_n = fem.Function(Q)
p_n.x.array[:] = 0.0

Tn = fem.Function(Th)
Tn.x.array[:] = T_init

Theta = ufl.TrialFunction(Th)
sT = ufl.TestFunction(Th)

# --------------------------------------------------
# VARIATIONAL FORMS
# --------------------------------------------------
gamma_div = PETSc.ScalarType(1e-8)
eta_brink = PETSc.ScalarType(1e-10)

a_mom = (
    (1.0 / dt) * ufl.inner(U, v) * dx
    + ufl.inner(ufl.grad(U) * u_n, v) * dx
    + 2 * nu * ufl.inner(ufl.sym(ufl.grad(U)), ufl.sym(ufl.grad(v))) * dx
    - ufl.inner(P, ufl.div(v)) * dx
    + ufl.inner(ufl.div(U), q) * dx
    + gamma_div * ufl.inner(ufl.div(U), ufl.div(v)) * dx
    + eta_brink * ufl.inner(U, v) * dx
)

L_mom = (
    (1.0 / dt) * ufl.inner(u_n, v) * dx
    - beta * ufl.inner((Tn - T_ref) * g_vec, v) * dx
)

a_T = (
    (1.0 / dt) * ufl.inner(Theta, sT) * dx
    + alpha * ufl.inner(ufl.grad(Theta), ufl.grad(sT)) * dx
    + ufl.inner(u_n, ufl.grad(Theta)) * sT * dx
)

L_T = (1.0 / dt) * ufl.inner(Tn, sT) * dx

opts_stokes = {
    "ksp_type": "gmres",
    "pc_type": "lu",
    "ksp_rtol": 1e-8,
}

opts_temp = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_rtol": 1e-10,
}

problem_NS = LinearProblem(
    a_mom,
    L_mom,
    bcs=[bc_u, bc_p0],
    petsc_options_prefix="ns_",
    petsc_options=opts_stokes,
)

problem_T = LinearProblem(
    a_T,
    L_T,
    bcs=[bc_T_bottom, bc_T_top],
    petsc_options_prefix="temp_",
    petsc_options=opts_temp,
)

# --------------------------------------------------
# OUTPUT BP / ADIOS2
# --------------------------------------------------
def write_viz(writer, u_field, p_field, T_field, time_value):
    u_viz.interpolate(u_field)
    p_viz.interpolate(p_field)
    T_viz.interpolate(T_field)

    u_viz.x.scatter_forward()
    p_viz.x.scatter_forward()
    T_viz.x.scatter_forward()

    writer.write(time_value)

TAG("TAG: start time stepping (fluid = Water, simple mesh)")

def classify_ra(Ra):
    if Ra < 1e6:
        return "laminar"
    if Ra < 1e8:
        return "transition"
    return "turbulent tendency"

# --------------------------------------------------
# TIME LOOP
# --------------------------------------------------
with VTXWriter(
    mesh.comm,
    "results_transient_simple_mesh.bp",
    [u_viz, p_viz, T_viz],
) as vtx:

    write_viz(vtx, u_n, p_n, Tn, 0.0)

    for step in range(1, nsteps + 1):

        t = step * dt

        # Solve Navier-Stokes + Boussinesq
        w_sol = problem_NS.solve()

        Uh, map_u = W.sub(0).collapse()
        Ph, map_p = W.sub(1).collapse()

        u_out = fem.Function(Uh)
        u_out.x.array[:] = w_sol.x.array[map_u]

        p_out = fem.Function(Ph)
        p_out.x.array[:] = w_sol.x.array[map_p]

        # Solve temperature
        T_out = problem_T.solve()

        # Update fields
        u_n.interpolate(u_out)
        p_n.interpolate(p_out)
        Tn.x.array[:] = T_out.x.array

        # Recreate problems because they depend on u_n and Tn
        a_mom = (
            (1.0 / dt) * ufl.inner(U, v) * dx
            + ufl.inner(ufl.grad(U) * u_n, v) * dx
            + 2 * nu * ufl.inner(ufl.sym(ufl.grad(U)), ufl.sym(ufl.grad(v))) * dx
            - ufl.inner(P, ufl.div(v)) * dx
            + ufl.inner(ufl.div(U), q) * dx
            + gamma_div * ufl.inner(ufl.div(U), ufl.div(v)) * dx
            + eta_brink * ufl.inner(U, v) * dx
        )

        L_mom = (
            (1.0 / dt) * ufl.inner(u_n, v) * dx
            - beta * ufl.inner((Tn - T_ref) * g_vec, v) * dx
        )

        problem_NS = LinearProblem(
            a_mom,
            L_mom,
            bcs=[bc_u, bc_p0],
            petsc_options_prefix=f"ns_{step}_",
            petsc_options=opts_stokes,
        )

        a_T = (
            (1.0 / dt) * ufl.inner(Theta, sT) * dx
            + alpha * ufl.inner(ufl.grad(Theta), ufl.grad(sT)) * dx
            + ufl.inner(u_n, ufl.grad(Theta)) * sT * dx
        )

        L_T = (1.0 / dt) * ufl.inner(Tn, sT) * dx

        problem_T = LinearProblem(
            a_T,
            L_T,
            bcs=[bc_T_bottom, bc_T_top],
            petsc_options_prefix=f"temp_{step}_",
            petsc_options=opts_temp,
        )

        if step % save_every == 0 or step == nsteps:
            write_viz(vtx, u_n, p_n, Tn, t)

            dT_cur = float(np.max(Tn.x.array) - np.min(Tn.x.array))
            U_est = float(np.sqrt(max(g * beta * dT_cur * H, 0.0)))
            Re_est = U_est * H / nu
            Ra_est = g * beta * dT_cur * H**3 / (nu * alpha)

            TAG(
                f"t={t:6.1f}s | "
                f"ΔT≈{dT_cur:6.2f} K | "
                f"U*≈{U_est:7.4f} m/s | "
                f"Re≈{Re_est:7.2f} | "
                f"Ra≈{Ra_est:8.2e} → {classify_ra(Ra_est)}"
            )

TAG("TAG: simulation completed → results_transient_simple_mesh.bp")