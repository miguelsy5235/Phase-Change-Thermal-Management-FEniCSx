import meshio
# ---------------------------------------------------
# Read Gmsh mesh
# ---------------------------------------------------
msh = meshio.read("battery_pcm_quad.msh")
# ---------------------------------------------------
# QUADRILATERAL CELLS
# ---------------------------------------------------
quad_cells = msh.get_cells_type("quad")
quad_data = msh.get_cell_data("gmsh:physical", "quad")

quad_mesh = meshio.Mesh(
    points=msh.points[:, :2],
    cells=[("quad", quad_cells)],
    cell_data={"name_to_read": [quad_data]},
)
# ---------------------------------------------------
# BOUNDARY LINES
# ---------------------------------------------------
line_cells = msh.get_cells_type("line")
line_data = msh.get_cell_data("gmsh:physical", "line")

line_mesh = meshio.Mesh(
    points=msh.points[:, :2],
    cells=[("line", line_cells)],
    cell_data={"name_to_read": [line_data]},
)
# ---------------------------------------------------
# WRITE XDMF FILES
# ---------------------------------------------------
meshio.write("mesh_quad.xdmf", quad_mesh)
meshio.write("mf_quad.xdmf", line_mesh)
print("Quadrilateral mesh conversion completed")
