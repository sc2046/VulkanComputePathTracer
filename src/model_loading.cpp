#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iostream>
#include <string>
#include <vector>


struct MeshData
{
	std::vector<float>		mVertices;
	std::vector<uint32_t>	mIndices;
};

MeshData LoadMeshFromObj(const std::string& path)
{

	std::string inputfile = "scenes/cornell_box.obj";
	tinyobj::ObjReaderConfig reader_config;
	reader_config.mtl_search_path = "./"; // Path to material files

	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(inputfile, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning();
	}

	// Get vertices of the mesh.
	const std::vector<tinyobj::real_t>   objVertices = reader.GetAttrib().GetVertices();

	const std::vector<tinyobj::shape_t>& objShapes = reader.GetShapes();  // All shapes in the file
	assert(objShapes.size() == 1);                                          // Check that this file has only one shape
	const tinyobj::shape_t& objShape = objShapes[0];                        // Get the first shape

	// Get the indices of the vertices of the first mesh of `objShape` in `attrib.vertices`:
	std::vector<uint32_t> objIndices;
	objIndices.reserve(objShape.mesh.indices.size());
	for (const tinyobj::index_t& index : objShape.mesh.indices)
	{
		objIndices.push_back(index.vertex_index);
	}

	return { objVertices, objIndices };
}
