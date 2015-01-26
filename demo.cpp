#include "../../DHX/dhx.hpp"

#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>

#include "script.hpp"

void test(){
  liris::BU3DMesh mesh;

  if ( !OpenMesh::IO::read_mesh(mesh,"/home/ding/BU3D/F0001/F0001_AN01WH_F3D.wrl"))
    {
      std::cerr << "Error loading mesh from file " << std::endl;
      return;
    }

  std::cout << mesh.information() << std::endl;

  // liris::huibin::Descriptor desc(mesh);

  //  std::cout<<"original point:  "<<mesh.point(Mesh::VertexHandle(0))<<std::endl;
}

int main(){
// test();
  //Script::run_preprocess( "/home/ding/BU3D",  "/home/ding/BU_Work/BU_step2_filling_hole");
  //Script::run_icp("/home/ding/BU_Work/BU_step2_filling_hole", "/home/ding/BU_Work/BU_step3_icp");
  // Script::tex_projection("/home/ding/BU_Work/BU_step3_icp", "/home/ding/BU3D", "/home/ding/BU_Work/BU_step4_texpro");
  //  Script::landmark_location("/home/ding/BU_Work/BU_step4_texpro", "/home/ding/BU_Work/BU_step4_texpro");
  //Script::back_projection("/home/ding/BU_Work/BU_step3_icp", "/home/ding/BU_Work/BU_step4_texpro", "/home/ding/BU_Work/BU_step4_texpro");
  //  Script::tex_projection("/home/ding/BU_Work/BU_step4_texpro", "/home/ding/BU3D", "/home/ding/BU_Work/BU_step4_texpro_test")
  // Script::FeatureExtration("/home/ding/BU_Work/BU_step4_texpro", "/home/ding/BU_Work/BU_step5_feature");
  //Script::GenerateExpeimentConfigration("/home/ding/BU_Work/BU_step4_texpro","/home/ding/BU_Work/BU_step5_feature", "/home/ding/BU_Work/BU_step6_Experiment");
  // Script::GenerateLibSVMFile("/home/ding/BU_Work/BU_step5_feature","/home/ding/BU_Work/BU_step6_Experiment", "/home/ding/BU_Work/BU_step7_libsvm");
  //Script::RunSVM("/home/ding/BU_Work/BU_step5_feature","/home/ding/BU_Work/BU_step6_Experiment");
  Script::RunAllSVM("/home/ding/BU_Work/BU_step5_feature","/home/ding/BU_Work/BU_step6_Experiment");
  return 0;
}
