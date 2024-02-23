add_executable(sub-mesh sub-mesh.cxx)
target_link_libraries(sub-mesh PUBLIC ${ITK_LIBRARIES})

add_test(NAME sub-mesh-help COMMAND sub-mesh --help)

add_test(NAME sub-mesh
  COMMAND sub-mesh
    ${CMAKE_CURRENT_SOURCE_DIR}/cow.vtk
    --cell-identifiers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    ${CMAKE_CURRENT_BINARY_DIR}/cow-sub.vtk
  )