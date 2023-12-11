# itkwasm-sub-mesh

Extract a subset of a mesh given by the cell identifiers.

## Installation

### Python

```sh
pip install itkwasm-sub-mesh
```

### JavaScript

```sh
npm install @itk-wasm/sub-mesh
```

## Usage

Python:

```python
# pip install itkwasm-mesh-io
from itkwasm_mesh_io import read_mesh

mesh = read_mesh('my_mesh.vtk')

from itkwasm_sub_mesh import sub_mesh

cell_identifiers = [0,3,4,5,9]
sub = sub_mesh(mesh, cell_identifiers)
```
