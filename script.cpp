#include "script.hpp"
#include "../../DHX/dhx.hpp"
#include "fusion.hpp"

#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace liris;

void Script::run_preprocess(const string& meshSrcPath, const string& dstPath){

  sys::makedir(dstPath);

  vector<std::string> persons = sys::dir(meshSrcPath);
  std::cout<<persons.size()<<std::endl;



  for(size_t i=0; i<100; ++i){
    string person_path = persons.at(i);
    std::cout<<person_path<<std::endl;
    string person_id = person_path.substr(person_path.find_last_of("/")+1, person_path.length());
    sys::makedir(dstPath + "/" + person_id);

    vector<string> scans = sys::dir(persons.at(i), "wrl");

    for (size_t j=0; j<scans.size(); ++j){
      string scan_file = scans.at(j);
      std::cout<<scan_file<<std::endl;
      string scan_id = scan_file.substr(scan_file.find_last_of("/") + 1, scan_file.find_last_of(".") - scan_file.find_last_of("/") -1);
      liris::BU3DMesh mesh;
      mesh.set_mesh_id(scan_id);
      mesh.set_need_to_be_fixed(true);
      mesh.set_preprocess_file("meta/bu3d_blacklist.xml");
      // std::cout<<mesh.get_name()<<std::endl;

      if ( !OpenMesh::IO::read_mesh(mesh,scan_file)){
        std::cerr << "Error loading mesh from file " << std::endl;
      }

      //std::cout << mesh.information() << std::endl<<std::endl<<std::endl;


      if ( !OpenMesh::IO::write_mesh( mesh, dstPath + "/" + person_id + "/" + mesh.mesh_id()+ ".mesh") ){
        std::cerr << "Error" << std::endl;
        std::cerr << "Possible reasons:\n";
        std::cerr << "1. Chosen format cannot handle an option!\n";
        std::cerr << "2. Mesh does not provide necessary information!\n";
        std::cerr << "3. Or simply cannot open file for writing!\n";
      }
      if ( !OpenMesh::IO::write_mesh( mesh, dstPath + "/" + person_id + "/" + mesh.mesh_id()+ ".obj") ){
        std::cerr << "Error" << std::endl;
        std::cerr << "Possible reasons:\n";
        std::cerr << "1. Chosen format cannot handle an option!\n";
        std::cerr << "2. Mesh does not provide necessary information!\n";
        std::cerr << "3. Or simply cannot open file for writing!\n";
      }
//  sleep(2);
    }
  }
}


void Script::run_icp(const string& meshSrcPath, const string& dstPath){

  sys::makedir(dstPath);

  vector<string> persons = sys::dir(meshSrcPath);
  std::cout<<persons.size()<<std::endl;



  string meshTemplateFile = meshSrcPath + "/F0014/F0014_NE00WH_F3D.mesh";
  liris::BU3DMesh mesh_template;

  if ( !OpenMesh::IO::read_mesh(mesh_template,meshTemplateFile)){
    std::cerr << "Error loading mesh from file " << std::endl;
  }

  for(size_t i=0; i<100; ++i){
    string person_path = persons.at(i);
    std::cout<<person_path<<std::endl;
    string person_id = person_path.substr(person_path.find_last_of("/")+1, person_path.length());
    sys::makedir(dstPath + "/" + person_id);

    vector<string> scans = sys::dir(persons.at(i), "mesh");

    for (size_t j=0; j<scans.size(); ++j){
      string scan_file = scans.at(j);
      string scan_id = scan_file.substr(scan_file.find_last_of("/") + 1, scan_file.find_last_of(".") - scan_file.find_last_of("/") -1);
      std::cout<<scan_file<<std::endl;

      liris::BU3DMesh mesh;
      mesh.set_mesh_id(scan_id);
      mesh.set_need_to_be_fixed(false);
      // std::cout<<mesh.get_name()<<std::endl;

      if ( !OpenMesh::IO::read_mesh(mesh,scan_file)){
        std::cerr << "Error loading mesh from file " << std::endl;
      }

      liris::MeshICP meshicp;
      meshicp.set_template(mesh_template);
      meshicp.run(mesh);

      if ( !OpenMesh::IO::write_mesh( mesh, dstPath + "/" + person_id + "/" + mesh.mesh_id()+ ".mesh") ){
        std::cerr << "Error" << std::endl;
        std::cerr << "Possible reasons:\n";
        std::cerr << "1. Chosen format cannot handle an option!\n";
        std::cerr << "2. Mesh does not provide necessary information!\n";
        std::cerr << "3. Or simply cannot open file for writing!\n";
      }
      if ( !OpenMesh::IO::write_mesh( mesh, dstPath + "/" + person_id + "/" + mesh.mesh_id()+ ".obj" ) ){
        std::cerr << "Error" << std::endl;
        std::cerr << "Possible reasons:\n";
        std::cerr << "1. Chosen format cannot handle an option!\n";
        std::cerr << "2. Mesh does not provide necessary information!\n";
        std::cerr << "3. Or simply cannot open file for writing!\n";
      }

    }
  }
}

void  Script::tex_projection(const string& meshSrcPath, const string& texSrcPath, const string& dstPath){
  sys::makedir(dstPath);

  vector<string> persons = sys::dir(meshSrcPath);
  cout<<persons.size()<<endl;

  //#pragma omp parallel for
  for(size_t i=0; i<100; ++i){
    string person_path = persons.at(i);
    cout<<person_path<<endl;
    string person_id = person_path.substr(person_path.find_last_of("/")+1, person_path.length());
    sys::makedir(dstPath + "/" + person_id);

    vector<string> scans = sys::dir(persons.at(i), "mesh");

    for (size_t j=0; j<scans.size(); ++j){
      string scan_file = scans.at(j);
      string scan_id = scan_file.substr(scan_file.find_last_of("/") + 1, scan_file.find_last_of(".") - scan_file.find_last_of("/") -1);

      string texture_file =  texSrcPath + "/" + person_id + "/" + scan_id+ ".bmp";
      cout<<scan_file<<"   "<<texture_file<<endl;

      BU3DMesh mesh(scan_id);

      if ( !OpenMesh::IO::read_mesh(mesh,scan_file)){
        std::cerr << "Error loading mesh from file " << std::endl;
      }

      cv::Mat tex = cv::imread(texture_file);
      mesh.set_texture(tex);

      if (mesh.landmark().dims>=2)
        cout<<"mesh has landmark: "<<mesh.landmark().size()<<endl;

      MeshProjection meshpro(mesh);
      MeshImage meshimage;
      meshpro.project(meshimage, 256, 320);

      cv::imwrite(dstPath + "/" + person_id + "/" + scan_id+ ".bmp",  meshimage.texture_image());
      if (meshimage.has_texture_landmark())
        cv::imwrite(dstPath + "/" + person_id + "/" + scan_id+ "_rePro.bmp",  meshimage.texture_image_marked());
    }
  }
}

void Script::landmark_location(const string& imgSrcPath, const string& dstPath){
  sys::makedir(dstPath);

  vector<string> persons = sys::dir(imgSrcPath);
  cout<<persons.size()<<endl;

   liris::FaceLandmark flm("shape_predictor_68_face_landmarks.dat");
   //#pragma omp parallel for
  for(size_t i=0; i<100; ++i){
    string person_path = persons.at(i);
    cout<<person_path<<endl;
    string person_id = person_path.substr(person_path.find_last_of("/")+1, person_path.length());
    sys::makedir(dstPath + "/" + person_id);

    vector<string> scans = sys::dir(persons.at(i), "3D.bmp");

    for (size_t j=0; j<scans.size(); ++j){
      string scan_file = scans.at(j);
      string scan_id = scan_file.substr(scan_file.find_last_of("/") + 1, scan_file.find_last_of(".") - scan_file.find_last_of("/") -1);
      string texture_file =  imgSrcPath + "/" + person_id + "/" + scan_id+ ".bmp";
      cout<<texture_file<<endl;
      cv::Mat img = cv::imread(texture_file);

      std::vector<cv::Mat> marks = flm.detect(img);
      //std::cout<<marks.at(0)<<std::endl;
      MeshImage meshimage;
      meshimage.set_texture_image(img);
      meshimage.set_texture_landmark(marks.at(0));

      string dst_texture_file =  imgSrcPath + "/" + person_id + "/" + scan_id+ "_marked.bmp";
      cv::imwrite(dst_texture_file, meshimage.texture_image_marked());

      string dst_texture_mark_file =  imgSrcPath + "/" + person_id + "/" + scan_id+ ".lm2d";
      meshimage.save_texture_landmark(dst_texture_mark_file);
      // sleep(5);
    }
  }
}

static cv::Mat read_landmark(const string& filename){
  vector<double> marks;
  double a;
  std::fstream f_in(filename.c_str(), std::ios::in);
  while (f_in>>a){
    marks.push_back(a);
  }
  f_in.close();
  return cv::Mat(marks, true).reshape(1,marks.size()/2);
}

void Script::back_projection(const string& meshSrcPath, const string& markSrcPath, const string & dstPath){
  sys::makedir(dstPath);

  vector<string> persons = sys::dir(meshSrcPath);
  cout<<persons.size()<<endl;

  //#pragma omp parallel for
  for(size_t i=0; i<100; ++i){
    string person_path = persons.at(i);
    cout<<person_path<<endl;
    string person_id = person_path.substr(person_path.find_last_of("/")+1, person_path.length());
    sys::makedir(dstPath + "/" + person_id);

    vector<string> scans = sys::dir(persons.at(i), "mesh");

    for (size_t j=0; j<scans.size(); ++j){
      string scan_file = scans.at(j);
      string scan_id =  scan_file.substr(scan_file.find_last_of("/") + 1, scan_file.find_last_of(".") - scan_file.find_last_of("/") -1);

      string landmark_file =  markSrcPath + "/" + person_id + "/" + scan_id+ ".lm2d";
      cv::Mat mark_2d = read_landmark(landmark_file);
      cout<<scan_file<<"   "<<landmark_file<<endl;
      cout<<"landmark size: "<<mark_2d.rows<<endl;

      BU3DMesh mesh(scan_id);

      if ( !OpenMesh::IO::read_mesh(mesh,scan_file)){
        std::cerr << "Error loading mesh from file " << std::endl;
      }

      MeshProjection meshpro(mesh);
      meshpro.back_project(mark_2d, 256, 320);
      //    cout<<mesh.getLandmark().size()<<endl;


      if ( !OpenMesh::IO::write_mesh( mesh, dstPath + "/" + person_id + "/" + mesh.mesh_id()+ ".mesh") ){
        std::cerr << "Error" << std::endl;
        std::cerr << "Possible reasons:\n";
        std::cerr << "1. Chosen format cannot handle an option!\n";
        std::cerr << "2. Mesh does not provide necessary information!\n";
        std::cerr << "3. Or simply cannot open file for writing!\n";
      }

      string texture_file =  markSrcPath + "/" + person_id + "/" + scan_id+ ".bmp";
      cout<<texture_file<<endl;
      cv::Mat img = cv::imread(texture_file);

      //std::cout<<marks.at(0)<<std::endl;
      MeshImage meshimage;
      meshimage.set_texture_image(img);
      meshimage.set_texture_landmark(mark_2d);

      string dst_texture_file =  markSrcPath + "/" + person_id + "/" + scan_id+ "_marked.bmp";
      cv::imwrite(dst_texture_file, meshimage.texture_image_marked());

    }
  }
}

void Script::FeatureExtration(const string& meshSrcPath, const string& dstPath){
  sys::makedir(dstPath);
  {
    cv::FileStorage fs(dstPath + "/bu_hog.xml",  cv::FileStorage::WRITE);
    fs.release();
  }

  {
    cv::FileStorage fs(dstPath + "/bu_hos.xml",  cv::FileStorage::WRITE);
    fs.release();
  }

  vector<string> persons = sys::dir(meshSrcPath);
  cout<<persons.size()<<endl;

  //#pragma omp parallel for
  for(size_t i=0; i<100; ++i){
    string person_path = persons.at(i);
    cout<<person_path<<endl;
    // string person_id = person_path.substr(person_path.find_last_of("/")+1, person_path.length());
    //   sys::makedir(dstPath + "/" + person_id);

    vector<string> scans = sys::dir(persons.at(i), "mesh");

    for (size_t j=0; j<scans.size(); ++j){
      string scan_file = scans.at(j);
      string scan_id =  scan_file.substr(scan_file.find_last_of("/") + 1, scan_file.find_last_of(".") - scan_file.find_last_of("/") -1);

      BU3DMesh mesh(scan_id);

      if ( !OpenMesh::IO::read_mesh(mesh,scan_file)){
        std::cerr << "Error loading mesh from file " << std::endl;
      }
      liris::huibin::Descriptor desc(mesh);
      {
        cv::FileStorage fs(dstPath + "/bu_hog.xml",  cv::FileStorage::APPEND);
        if (!cv::checkRange(desc.HoG())){
          cout<<"HoG "<<scan_id<<endl;
        }
        fs<<scan_id<<desc.HoG();
        fs.release();
      }
      {
        cv::FileStorage fs(dstPath + "/bu_hos.xml",  cv::FileStorage::APPEND);
        if (!cv::checkRange(desc.HoS())){
          cout<<"HOS "<<scan_id<<endl;
        }
        fs<<scan_id<<desc.HoS();
        fs.release();
      }
    }
  }
}

static string int2str(int i){
  std::stringstream ss;
  ss<<i;
  return ss.str();
}


void Script::GenerateExpeimentConfigration(const string& meshSrcPath, const string& featureSrcPath, const string& dstPath){
  sys::makedir(dstPath);

  vector<string> persons = sys::dir(meshSrcPath);
  cout<<persons.size()<<endl;

  //#pragma omp parallel for
  vector<vector<string> > samples(persons.size());
  vector<vector<int> > labels(persons.size());
  for(size_t i=0; i<100; ++i){
    string person_path = persons.at(i);
    cout<<person_path<<endl;
    // string person_id = person_path.substr(person_path.find_last_of("/")+1, person_path.length());
    //   sys::makedir(dstPath + "/" + person_id);
    vector<string> scans = sys::dir(persons.at(i), "mesh");
    for (size_t j=0; j<scans.size(); ++j){
      string scan_file = scans.at(j);
      string scan_id =  scan_file.substr(scan_file.find_last_of("/") + 1, scan_file.find_last_of(".") - scan_file.find_last_of("/") -1);

      if (scan_id.find("AN03") != string::npos ||
          scan_id.find("AN04") != string::npos ||
          scan_id.find("DI03") != string::npos ||
          scan_id.find("DI04") != string::npos ||
          scan_id.find("FE03") != string::npos ||
          scan_id.find("FE04") != string::npos ||
          scan_id.find("HA03") != string::npos ||
          scan_id.find("HA04") != string::npos ||
          scan_id.find("SA03") != string::npos ||
          scan_id.find("SA04") != string::npos ||
          scan_id.find("SU03") != string::npos ||
          scan_id.find("SU04") != string::npos
          ){
        samples.at(i).push_back(scan_id);
        if (scan_id.find("AN") != string::npos){
          labels.at(i).push_back(1);
        }else if (scan_id.find("DI") != string::npos){
          labels.at(i).push_back(2);
        }else if (scan_id.find("FE") != string::npos){
          labels.at(i).push_back(3);
        }else if (scan_id.find("HA") != string::npos){
          labels.at(i).push_back(4);
        }else if (scan_id.find("SA") != string::npos){
          labels.at(i).push_back(5);
        }else if (scan_id.find("SU") != string::npos){
          labels.at(i).push_back(6);
        }
      }
    }
  }

  vector<int> random_v;
  for (int i=0; i<100; ++i)
    random_v.push_back(i);

  cv::FileStorage fs(featureSrcPath +  "/bu_hog.xml", cv::FileStorage::READ );
  for (int i=0; i<1000; ++i){
    std::random_shuffle(random_v.begin(), random_v.end());
    string name = dstPath + "/Exp_" + int2str(i);
    std::fstream f_out(name.c_str(), std::ios::out);

    for (size_t j=0; j<54; ++j){
      for(int k=0; k<12; ++k){
        f_out<<samples.at(random_v.at(j)).at(k)<<endl;
      }
    }

    for (size_t j=54; j<60; ++j){
      for(int k=0; k<12; ++k){
        f_out<<samples.at(random_v.at(j)).at(k)<<endl;
      }
    }


    f_out.close();
  }
  fs.release();
}


void Script::GenerateLibSVMFile(const string& featureSrcPath, const string& expSrcPath, const string& dstPath){
  sys::makedir(dstPath);
  sys::makedir(dstPath+ "/hos");
  cv::FileStorage fs(featureSrcPath + "/bu_hos.xml",  cv::FileStorage::READ);

  vector<string> exps = sys::dir(expSrcPath);
  for (size_t i=0; i<10; ++i){
    string exp_file = exps.at(i);
    string name;
    vector<string> samples;
    std::fstream f_in(exp_file.c_str(), std::ios::in);
    while(f_in>>name){
      samples.push_back(name);
    }
    f_in.close();

    {
      string out_name = dstPath + "/hos/train_" + int2str(i);
      std::fstream  f_out(out_name.c_str(), std::ios::out);
      for (size_t k=0; k<54*12; k++){
        name = samples.at(k);
        if (name.find("AN") != string::npos){
          f_out<<"1";
        }else  if (name.find("DI") != string::npos){
          f_out<<"2";
        }else  if (name.find("FE") != string::npos){
          f_out<<"3";
        }else  if (name.find("HA") != string::npos){
          f_out<<"4";
        }else  if (name.find("SA") != string::npos){
          f_out<<"5";
        }else  if (name.find("SU") != string::npos){
          f_out<<"6";
        }
        cv::Mat mat;
        fs[name]>>mat;
        for (int i=0; i<mat.cols; ++i)
          f_out<<" "<<i+1<<":"<<mat.at<double>(0,i);
        f_out<<endl;
      }
      f_out.close();
    }

    {
      string out_name = dstPath + "/hos/test_" + int2str(i);
      std::fstream  f_out(out_name.c_str(), std::ios::out);
      for (size_t k=54*12; k<60*12; k++){
        name = samples.at(k);
        if (name.find("AN") != string::npos){
          f_out<<"1";
        }else  if (name.find("DI") != string::npos){
          f_out<<"2";
        }else  if (name.find("FE") != string::npos){
          f_out<<"3";
        }else  if (name.find("HA") != string::npos){
          f_out<<"4";
        }else  if (name.find("SA") != string::npos){
          f_out<<"5";
        }else  if (name.find("SU") != string::npos){
          f_out<<"6";
        }
        cv::Mat mat;
        fs[name]>>mat;
        for (int i=0; i<mat.cols; ++i)
          f_out<<" "<<i+1<<":"<<mat.at<double>(0,i);
        f_out<<endl;
      }
      f_out.close();
    }
  }
  fs.release();
}


static void static_analysis(std::vector<double>& v, double& m, double& stdev){
  for (size_t i=0; i<v.size(); ++i){
    cout<<v.at(i)<<"  ";
  }
  cout<<endl;

  double sum = 0;
  for (size_t i=0; i<v.size(); ++i)
    sum += v.at(i);
  m =  sum / v.size();

  double accum = 0.0;
  for (size_t i=0; i<v.size(); ++i)
    accum += (v.at(i)- m) * (v.at(i) - m);

  stdev = sqrt(accum / v.size());
}

void Script::RunSVM(const string& featureSrcPath, const string& expSrcPath){


  vector<string> exps = sys::dir(expSrcPath);
  vector<double> mean, stddev, cc, gg;
  for (int c = 1; c<9; c=c+2){
    for (int g=-1; g>-15; g=g-2){
      double c_2 = pow(2, c);
      double g_2 = pow(2, g);
      cout<<c_2<<"   "<<g_2<<endl;
      cc.push_back(c_2);
      gg.push_back(g_2);

      int exp_num  = 10;

      vector<double> rates(exp_num);
      #pragma omp parallel for
      for (int i=0; i<exp_num; ++i){

        cv::FileStorage fs(featureSrcPath + "/bu_hos.xml",  cv::FileStorage::READ);

        string exp_file = exps.at(i);
        string name;
        vector<string> samples;
        std::fstream f_in(exp_file.c_str(), std::ios::in);
        while(f_in>>name){
          samples.push_back(name);
        }
        f_in.close();

        SVMParameter svm_params;
        svm_params.set_kernel_type(SVMParameter::SVM_RBF);
        svm_params.set_C(c_2);
        svm_params.set_gamma(g_2);
        SVMAdapter svm(svm_params);

        {
          cv::Mat train_data(54*12, 3528, CV_64FC1);
          cv::Mat train_label(54*12, 1, CV_32SC1);
          for (size_t k=0; k<54*12; k++){
            name = samples.at(k);
            if (name.find("AN") != string::npos){
              train_label.at<int>(k,0) = 1;
            }else  if (name.find("DI") != string::npos){
              train_label.at<int>(k,0) = 2;
            }else  if (name.find("FE") != string::npos){
              train_label.at<int>(k,0) = 3;
            }else  if (name.find("HA") != string::npos){
              train_label.at<int>(k,0) = 4;
            }else  if (name.find("SA") != string::npos){
              train_label.at<int>(k,0) = 5;
            }else  if (name.find("SU") != string::npos){
              train_label.at<int>(k,0) = 6;
            }
            cv::Mat mat;
            fs[name]>>mat;
            mat.copyTo(train_data.row(k));
          }
          svm.Train(train_data, train_label);
        }

        {
          cv::Mat test_data(6*12, 3528, CV_64FC1);
          cv::Mat test_label(6*12, 1, CV_32SC1);
          for (size_t k=54*12; k<60*12; k++){
            name = samples.at(k);
            if (name.find("AN") != string::npos){
              test_label.at<int>(k-54*12,0) = 1;
            }else  if (name.find("DI") != string::npos){
              test_label.at<int>(k-54*12,0) = 2;
            }else  if (name.find("FE") != string::npos){
              test_label.at<int>(k-54*12,0) = 3;
            }else  if (name.find("HA") != string::npos){
              test_label.at<int>(k-54*12,0) = 4;
            }else  if (name.find("SA") != string::npos){
              test_label.at<int>(k-54*12,0) = 5;
            }else  if (name.find("SU") != string::npos){
              test_label.at<int>(k-54*12,0) = 6;
            }
            cv::Mat mat;
            fs[name]>>mat;
            mat.copyTo(test_data.row(k-54*12));
          }
          cv::Mat label, prob;
          svm.Predict(test_data,label,prob);

          int error_num =0;
          for (int  kk=0; kk<test_label.rows; ++kk){
            if (test_label.at<int>(kk,0) != (int)label.at<double>(kk,0))
              error_num++;
          }
          // cout<<"Rate : "<<1- (double)error_num/(double)test_label.rows<<endl;
          rates.at(i) = 1- (double)error_num/(double)test_label.rows;
        }
        fs.release();
      }


      double a,b;
      static_analysis(rates,a, b);
      cout<<"Mean "<<a<<" STD"<<b<<endl;
      mean.push_back(a);
      stddev.push_back(b);
    }
  }

  int idx = std::max_element(mean.begin(), mean.end()) - mean.begin();
  cout<<"Max Mean:  c " <<cc.at(idx)<<"   g "<<gg.at(idx)<<"   Mean "<<mean.at(idx)<<"   Std "<<stddev.at(idx)<<endl;
  idx = std::min_element(stddev.begin(), stddev.end()) - stddev.begin();
  cout<<"Min STD :  c " <<cc.at(idx)<<"   g "<<gg.at(idx)<<"   Mean "<<mean.at(idx)<<"   Std "<<stddev.at(idx)<<endl;
}


void Script::RunAllSVM(const string& featureSrcPath, const string& expSrcPath){

  vector<string> exps = sys::dir(expSrcPath);

  int exp_num  = 10;

  //#pragma omp parallel for
  for (int i=30; i<exp_num+30; ++i){

  cv::FileStorage fs_hog(featureSrcPath + "/bu_hog.xml",  cv::FileStorage::READ);
  cv::FileStorage fs_hos(featureSrcPath + "/bu_hos.xml",  cv::FileStorage::READ);

  string exp_file = exps.at(i);
  string name;
  vector<string> samples;
  std::fstream f_in(exp_file.c_str(), std::ios::in);
  while(f_in>>name){
    samples.push_back(name);
  }
  f_in.close();

  Fusion fusion;
  {
    SVMParameter svm_params;
    svm_params.set_kernel_type(SVMParameter::SVM_RBF);
    svm_params.set_C(8);
    svm_params.set_gamma(0.0078125);
    fusion.set_params_hog(svm_params);
  }

  {
    SVMParameter svm_params;
    svm_params.set_kernel_type(SVMParameter::SVM_RBF);
    svm_params.set_C(32);
    svm_params.set_gamma(0.00195312);
    fusion.set_params_hos(svm_params);
  }

  {
    SVMParameter svm_params;
    svm_params.set_kernel_type(SVMParameter::SVM_LINEAR);
    fusion.set_params_hog_hos(svm_params);
  }


  {
    cv::Mat train_hog(54*12, 3528, CV_64FC1);
    cv::Mat train_hos(54*12, 3528, CV_64FC1);
    cv::Mat train_label(54*12, 1, CV_32SC1);
    for (size_t k=0; k<54*12; k++){
      name = samples.at(k);
      if (name.find("AN") != string::npos){
        train_label.at<int>(k,0) = 1;
      }else  if (name.find("DI") != string::npos){
        train_label.at<int>(k,0) = 2;
      }else  if (name.find("FE") != string::npos){
        train_label.at<int>(k,0) = 3;
      }else  if (name.find("HA") != string::npos){
        train_label.at<int>(k,0) = 4;
      }else  if (name.find("SA") != string::npos){
        train_label.at<int>(k,0) = 5;
      }else  if (name.find("SU") != string::npos){
        train_label.at<int>(k,0) = 6;
      }
      cv::Mat hog, hos;
      fs_hog[name]>>hog;
      fs_hos[name]>>hos;
      hog.copyTo(train_hog.row(k));
      hos.copyTo(train_hos.row(k));
    }
    fusion.Train(train_hog, train_hos,  train_label);
  }

  {
    cv::Mat test_hog(6*12, 3528, CV_64FC1);
    cv::Mat test_hos(6*12, 3528, CV_64FC1);
    cv::Mat test_label(6*12, 1, CV_32SC1);
    for (size_t k=54*12; k<60*12; k++){
      name = samples.at(k);
      if (name.find("AN") != string::npos){
        test_label.at<int>(k-54*12,0) = 1;
      }else  if (name.find("DI") != string::npos){
        test_label.at<int>(k-54*12,0) = 2;
      }else  if (name.find("FE") != string::npos){
        test_label.at<int>(k-54*12,0) = 3;
      }else  if (name.find("HA") != string::npos){
        test_label.at<int>(k-54*12,0) = 4;
      }else  if (name.find("SA") != string::npos){
        test_label.at<int>(k-54*12,0) = 5;
      }else  if (name.find("SU") != string::npos){
        test_label.at<int>(k-54*12,0) = 6;
      }
      cv::Mat hog, hos;
      fs_hog[name]>>hog;
      fs_hos[name]>>hos;
      hog.copyTo(test_hog.row(k-54*12));
      hos.copyTo(test_hos.row(k-54*12));
    }
    fusion.Predict(test_hog, test_hos,  test_label);
  }
  cout<<fusion.score_level()<<fusion.feature_level()<<endl;
  cout<<fusion.score_rate()<<"    "<<fusion.feature_rate()<<std::endl;
  fs_hog.release();
  fs_hos.release();
  }
}
