cmake_minimum_required(VERSION 3.16)
project(itkwasm-sub-mesh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(io_components)
if(NOT EMSCRIPTEN)
  set(io_components ITKIOMeshVTK)
endif()
find_package(ITK REQUIRED
  COMPONENTS
    ${io_components}
    WebAssemblyInterface
    ITKMesh
  )
include(${ITK_USE_FILE})

enable_testing()

# Begin create-itk-wasm added pipelines.
add_subdirectory(sub-mesh)
# End create-itk-wasm added pipelines.
