#ifndef __SCRIPT_HPP__
#define __SCRIPT_HPP__

#include <string>

using std::string;

class Script{
public:
  static void run_preprocess(const string& meshSrcPath, const string& dstPath);
  static void run_icp(const string& meshSrcPath, const string& dstPath);
  static void tex_projection(const string& meshSrcPath, const string& texSrcPath, const string & dstPath);
  static void landmark_location(const string& imgSrcPath, const string& dstPath);
  static void back_projection(const string& meshSrcPath, const string& markSrcPath, const string & dstPath);
  static void FeatureExtration(const string& meshSrcPath, const string& dstPath);
  static void GenerateExpeimentConfigration(const string& meshSrcPath,  const string& featureSrcPath, const string& dstPath);
  static void GenerateLibSVMFile(const string& featureSrcPath, const string& expSrcPath, const string& dstPath);
  static void RunSVM(const string& featureSrcPath, const string& expSrcPath);
  static void RunAllSVM(const string& featureSrcPath, const string& expSrcPath);
};

#endif
