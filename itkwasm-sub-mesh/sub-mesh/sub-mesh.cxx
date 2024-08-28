/*=========================================================================

 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "itkPipeline.h"
#include "itkInputMesh.h"
#include "itkInputTextStream.h"
#include "itkOutputMesh.h"
#include "itkSupportInputMeshTypes.h"

#include "itkVertexCell.h"
#include "itkLineCell.h"
#include "itkTriangleCell.h"
#include "itkQuadrilateralCell.h"
#include "itkTetrahedronCell.h"
#include "itkHexahedronCell.h"
#include "itkPolygonCell.h"

#include "itkCommonEnums.h"
#include "itkAutomaticTopologyMeshSource.h"
#include <unordered_map>

template <typename TMesh>
int subMesh(itk::wasm::Pipeline &pipeline, const TMesh *mesh)
{
  using MeshType = TMesh;
  constexpr unsigned int Dimension = MeshType::PointDimension;
  using PixelType = typename MeshType::PixelType;

  pipeline.get_option("mesh")->required()->type_name("INPUT_MESH");

  std::vector< uint32_t > cellIdentifiers;
  pipeline.add_option("--cell-identifiers", cellIdentifiers, "Cell identifiers for output mesh.")->required();

  itk::wasm::OutputMesh<MeshType> subMesh;
  pipeline.add_option("sub-mesh", subMesh, "Sub mesh.")->required()->type_name("OUTPUT_MESH");

  ITK_WASM_PARSE(pipeline);

  using CellAutoPointer = typename MeshType::CellAutoPointer;
  using CellType = typename MeshType::CellType;
  using VertexType = itk::VertexCell< CellType >;
  using LineType = itk::LineCell< CellType >;
  using TriangleType = itk::TriangleCell< CellType >;
  using QuadrilateralType = itk::QuadrilateralCell< CellType >;
  using TetrahedronType = itk::TetrahedronCell< CellType >;
  using HexahedronType = itk::HexahedronCell< CellType >;
  using PolygonType = itk::PolygonCell< CellType >;
  using CellGeometryEnum = itk::CommonEnums::CellGeometry;

  using PointIdIterator = typename CellType::PointIdConstIterator;
  std::set< itk::IdentifierType > usedPoints;
  for (auto cellIdentifier : cellIdentifiers)
  {
    CellAutoPointer cell;
    mesh->GetCell(cellIdentifier, cell);
    using PointIdIterator = typename CellType::PointIdConstIterator;
    PointIdIterator pointIdIterator = cell->PointIdsBegin();
    PointIdIterator pointIdEnd = cell->PointIdsEnd();
    while (pointIdIterator != pointIdEnd)
    {
      usedPoints.insert(*pointIdIterator);
      ++pointIdIterator;
    }
  }

  using MeshSourceType = itk::AutomaticTopologyMeshSource< MeshType >;
  auto meshSource = MeshSourceType::New();
  using OldNewPointMapType = std::unordered_map< itk::IdentifierType, itk::IdentifierType >;
  auto oldNewPointMap = OldNewPointMapType();
  itk::IdentifierType newPointId = 0;
  for (auto pointId : usedPoints)
  {
    meshSource->AddPoint(mesh->GetPoint(pointId));
    oldNewPointMap[pointId] = newPointId;
    newPointId++;
  }

  for (auto cellIdentifier : cellIdentifiers)
  {
    CellAutoPointer cell;
    mesh->GetCell(cellIdentifier, cell);
    using CellPointsType = itk::Array< itk::IdentifierType >;
    CellPointsType cellPoints(cell->GetNumberOfPoints());
    PointIdIterator pointIdIterator = cell->PointIdsBegin();
    for (unsigned int i = 0; i < cell->GetNumberOfPoints(); i++)
    {
      cellPoints[i] = oldNewPointMap[*pointIdIterator];
      ++pointIdIterator;
    }
    switch (cell->GetType())
    {
      case CellGeometryEnum::VERTEX_CELL:
      {
        meshSource->AddVertex(cellPoints);
        break;
      }
      case CellGeometryEnum::LINE_CELL:
      {
        meshSource->AddLine(cellPoints);
        break;
      }
      case CellGeometryEnum::TRIANGLE_CELL:
      {
        meshSource->AddTriangle(cellPoints);
        break;
      }
      case CellGeometryEnum::QUADRILATERAL_CELL:
      {
        meshSource->AddQuadrilateral(cellPoints);
        break;
      }
      case CellGeometryEnum::TETRAHEDRON_CELL:
      {
        meshSource->AddTetrahedron(cellPoints);
        break;
      }
      case CellGeometryEnum::POLYGON_CELL:
      {
        switch (cell->GetNumberOfPoints())
        {
          case 3:
          {
            meshSource->AddTriangle(cellPoints);
            break;
          }
          case 4:
          {
            meshSource->AddQuadrilateral(cellPoints);
            break;
          }
          default:
          {
            std::cerr << "Polygon cell with " << cell->GetNumberOfPoints() << " points not supported." << std::endl;
            return EXIT_FAILURE;
          }
        }
        break;
      }
      default:
      {
        std::cerr << "Cell type not supported." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  ITK_WASM_CATCH_EXCEPTION(pipeline, meshSource->Update());
  typename MeshType::ConstPointer result = meshSource->GetOutput();
  subMesh.Set(result);

  return EXIT_SUCCESS;
}

template <typename TMesh>
class PipelineFunctor
{
public:
  int operator()(itk::wasm::Pipeline &pipeline)
  {
    using MeshType = TMesh;

    itk::wasm::InputMesh<MeshType> mesh;
    pipeline.add_option("mesh", mesh, "Full mesh")->type_name("INPUT_MESH");

    ITK_WASM_PRE_PARSE(pipeline);

    typename MeshType::ConstPointer meshRef = mesh.Get();
    return subMesh<MeshType>(pipeline, meshRef);
  }
};

int main(int argc, char * argv[])
{
  itk::wasm::Pipeline pipeline("sub-mesh", "Extract a subset of a mesh given by the cell identifiers.", argc, argv);

  return itk::wasm::SupportInputMeshTypes<PipelineFunctor,
    uint8_t, uint16_t, int16_t, float, double>
    ::Dimensions<3U>("mesh", pipeline);
}
