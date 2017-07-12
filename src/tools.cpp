#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  if(estimations.size() != ground_truth.size() || estimations.size() == 0)
      return rmse;

  //accumulate squared residuals
  VectorXd error;
  for(int i=0; i < estimations.size(); ++i){
      error = estimations[i]-ground_truth[i];
      error = error.array()*error.array();
      rmse = rmse + error;
  }

std::cout<<"RMSE"<<std::endl;
std::cout<<estimations[estimations.size()-1]<<std::endl;
std::cout<<ground_truth[estimations.size()-1]<<std::endl;
std::cout<<"RMSE"<<std::endl;

  //calculate the mean
  rmse = rmse / estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}
