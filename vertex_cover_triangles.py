# Modified version of https://colab.research.google.com/drive/1oy-qBX32gs5-RHs-4kgkT3czT4JZv9RN
# Original version by Rasmus Pagh
from typing import List, Any

# Import APX wrapper class
import apx
from importlib import reload
reload(apx)
from apx import DataFile, LinearProgram, np
import pandas as pd


"""# Approximating vertex cover"""

Graph = List[List[Any]]


def get_triangles(graph: Graph):
  # Build adjacency list
  adjacency_list = {}
  for (u, v) in graph:
    neighbors = adjacency_list.get(u, set())
    neighbors.add(v)
    adjacency_list[u] = neighbors

    neighbors = adjacency_list.get(v, set())
    neighbors.add(u)
    adjacency_list[v] = neighbors

  triangles = set()

  # Iterate over all wedges
  for vertex1, neighbors1 in adjacency_list.items():
    for vertex2 in neighbors1:
      if vertex1 == vertex2:
        continue

      for vertex3 in adjacency_list.get(vertex2, set()):
        if vertex1 >= vertex3 or vertex2 == vertex3:
          continue

        # Check if wedge is closed
        if vertex1 in adjacency_list.get(vertex3, set()):
          triangles.add(tuple(sorted((vertex1, vertex2, vertex3))))

  return triangles


def get_basic_lp(graph: Graph):
  vertex_cover_lp = LinearProgram('min')
  objective = {}
  for (u,v) in graph:
    vertex_cover_lp.add_constraint({u: 1, v: 1}, 1)
    objective[u] = 1.0
    objective[v] = 1.0

  return vertex_cover_lp, objective


def get_triangle_lp(graph: Graph):
  vertex_cover_lp, objective = get_basic_lp(graph)

  for u, v, y in get_triangles(graph):
    vertex_cover_lp.add_constraint({u: 1, v: 1, y: 1}, 2)

  return vertex_cover_lp, objective


def evaluate_lp(lp_factory, filename):
  graph = DataFile(filename)
  graph = list(graph)

  vertex_cover_lp, objective = lp_factory(graph)

  vertex_cover_lp.set_objective(objective)

  value, solution = vertex_cover_lp.solve()
  rounded_value, rounded_solution = 0, {}
  for x in solution:
    r = int(np.round(solution[x] + 1e-10))
    # Add small constant to deal with numerical issues for numbers close to 1/2
    rounded_solution[x] = r
    rounded_value += r

  return value, solution, rounded_value, rounded_solution


results = pd.DataFrame()
diffs = pd.DataFrame()

lp_factories = [
  get_basic_lp,
  get_triangle_lp
]


for filename in DataFile.graph_files:
  diff_dict = {"filename": [filename]}

  for lp_factory in lp_factories:
    value, solution, rounded_value, rounded_solution = evaluate_lp(lp_factory, filename)
    results = pd.concat((results, pd.DataFrame({"filename": [filename], "lp_factory": [lp_factory.__name__], "value": [value], "solution": [solution], "rounded_value": [rounded_value], "rounded_solution": [rounded_solution]})), ignore_index=True)
    diff_dict[lp_factory.__name__ + "_value"] = [value]
    diff_dict[lp_factory.__name__ + "_rounded_value"] = [rounded_value]

  diffs = pd.concat((diffs, pd.DataFrame(diff_dict)), ignore_index=True)


diffs["value_increase"] = diffs["get_triangle_lp_value"] - diffs["get_basic_lp_value"]
diffs["rounded_value_decrease"] = diffs["get_basic_lp_rounded_value"] - diffs["get_triangle_lp_rounded_value"]
diffs["basic_approximation_ratio"] = diffs["get_basic_lp_rounded_value"] / diffs["get_basic_lp_value"]
diffs["triangle_approximation_ratio"] = diffs["get_triangle_lp_rounded_value"] / diffs["get_basic_lp_value"]

print(diffs[[
  "filename",
  "get_basic_lp_value",
  "get_triangle_lp_value",
  "get_basic_lp_rounded_value",
  "get_triangle_lp_rounded_value",
  "basic_approximation_ratio",
  "triangle_approximation_ratio"
]].style.format(precision=2).to_latex())
