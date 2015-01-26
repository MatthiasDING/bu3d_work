#include "fusion.hpp"
#include <iostream>

void Fusion::Train(cv::Mat&  hog, cv::Mat& hos, cv::Mat& label){
  svm_hog_ = new liris::SVMAdapter(svm_params_hog_);
  svm_hos_ = new liris::SVMAdapter(svm_params_hos_);
  svm_hog_hos_ = new liris::SVMAdapter(svm_params_hog_hos_);

  svm_hog_->Train(hog, label);
  svm_hos_->Train(hos, label);

  cv::Mat hog_hos(hog.rows, hog.cols*2, CV_64FC1);
  hog.copyTo(hog_hos(cv::Range::all(), cv::Range(0,hog.cols)));
  hos.copyTo(hog_hos(cv::Range::all(), cv::Range(hog.cols,hog.cols*2)));

  svm_hog_hos_->Train(hog_hos, label);
}


void Fusion::Predict(cv::Mat&  hog, cv::Mat& hos, cv::Mat& label){

  cv::Mat hog_predict, hog_prob;
  svm_hog_->Predict(hog, hog_predict, hog_prob);

  cv::Mat hos_predict, hos_prob;
  svm_hos_->Predict(hos, hos_predict, hos_prob);


  cv::Mat hog_hos(hog.rows, hog.cols*2, CV_64FC1);
  hog.copyTo(hog_hos(cv::Range::all(), cv::Range(0,hog.cols)));
  hos.copyTo(hog_hos(cv::Range::all(), cv::Range(hog.cols,hog.cols*2)));

  cv::Mat hog_hos_predict, hog_hos_prob;
  svm_hog_hos_->Predict(hog_hos, hog_hos_predict, hog_hos_prob);

  cv::Mat score_fusion_prob = hog_prob + hos_prob;

  hog_ = compute_rate(hog_prob, label);
  hos_ = compute_rate(hos_prob, label);
  score_level_ = compute_rate(score_fusion_prob, label);
  feature_level_ = compute_rate(hog_hos_prob, label);
}

cv::Mat Fusion::compute_rate(cv::Mat& prob, cv::Mat& label){
  cv::Mat rate = cv::Mat::zeros(6,6, CV_64FC1);
  for (int i=0; i<prob.rows; ++i){
    double max_prob = 0;
    int min_idx = -1;
    for (int j=0; j<prob.cols; ++j){
      if (prob.at<double>(i,j) > max_prob){
        max_prob = prob.at<double>(i,j);
        min_idx = j;
      }
    }
    rate.at<double>(label.at<int>(i,0)-1, min_idx) += 1;
  }
  rate  = rate/(label.rows/6);
  return rate;
}

double Fusion::overall(cv::Mat& rate){
  double r = 0;
  for (int i=0; i<rate.rows; ++i){
    r += rate.at<double>(i,i);
  }
  return r/rate.rows;
}

void Fusion::save(const std::string& filename){
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs<<"hog"<<hog_;
  fs<<"hos"<<hos_;
  fs<<"score"<<score_level_;
  fs<<"feature"<<feature_level_;
  fs.release();
}
