#ifndef __FUSION_HPP__
#define __FUSION_HPP__

#include <opencv2/opencv.hpp>
#include "../../DHX/dhx.hpp"
#include <string>

class Fusion{
public:
  Fusion():svm_hog_(0), svm_hos_(0), svm_hog_hos_(0){}
  ~Fusion(){
    if (svm_hog_)
      delete svm_hog_;
    if (svm_hos_)
      delete svm_hos_;
    if (svm_hog_hos_)
      delete svm_hog_hos_;
  }
  void Train(cv::Mat& hog, cv::Mat& hos, cv::Mat& label);
  void Predict(cv::Mat& hog, cv::Mat& hos, cv::Mat& label);

  void set_params_hog(const liris::SVMParameter& svm_params_hog){
    svm_params_hog_ = svm_params_hog;
  }

  void set_params_hos(const liris::SVMParameter& svm_params_hos){
    svm_params_hos_ = svm_params_hos;
  }

  void set_params_hog_hos(const liris::SVMParameter& svm_params_hog_hos){
    svm_params_hog_hos_ = svm_params_hog_hos;
  }

  cv::Mat score_level() const{
    return score_level_;
  }

  cv::Mat feature_level() const{
    return feature_level_;
  }

  cv::Mat hog() const{
    return hog_;
  }

  cv::Mat hos() const{
    return hos_;
  }

  double score_rate(){
    return overall(score_level_);
  }

  double feature_rate(){
    return overall(feature_level_);
  }

  double hog_rate(){
    return overall(hog_);
  }

  double hos_rate(){
    return overall(hos_);
  }

  void save(const std::string& filename);
private:
  cv::Mat compute_rate(cv::Mat& prob, cv::Mat& label);
  double overall(cv::Mat& rate);
  liris::SVMParameter svm_params_hog_, svm_params_hos_, svm_params_hog_hos_;
  liris::SVMAdapter *svm_hog_, *svm_hos_, *svm_hog_hos_;
  cv::Mat score_level_, feature_level_, hog_, hos_;
};

#endif
