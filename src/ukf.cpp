#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  n_x_ = 5;

  n_aug_ = 7;

  lambda_ = 3 - n_x_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd(2 * n_aug_ + 1);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_) {

    // first measurement

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      float vx = meas_package.raw_measurements_(2) * cos(meas_package.raw_measurements_(1));
      float vy = meas_package.raw_measurements_(2) * sin(meas_package.raw_measurements_(1));
      x_ << meas_package.raw_measurements_[0]*cos(meas_package.raw_measurements_[1]),
            meas_package.raw_measurements_[0]*sin(meas_package.raw_measurements_[1]),
            0, 0, 0;
      P_ << std_laspx_*std_laspx_,0,0,0,0,
            0,std_laspy_*std_laspy_,0,0,0,
            0,0,1,0,0,
            0,0,0,1,0,
            0,0,0,0,1;
      time_us_ = meas_package.timestamp_;
      cout<<"Radar"<<endl;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            0, 0, 0;
      P_ << std_laspx_*std_laspx_,0,0,0,0,
            0,std_laspy_*std_laspy_,0,0,0,
            0,0,1,0,0,
            0,0,0,1,0,
            0,0,0,0,1;
      time_us_ = meas_package.timestamp_;
      cout<<"Laser"<<endl;
    }

    is_initialized_ = true;
    return;
  }

  float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)  {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  VectorXd x_aug_ = VectorXd(n_aug_);
  MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  //create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5,5) = P_;
  P_aug_(5,5) = std_a_*std_a_;
  P_aug_(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0) << x_aug_;
  //std::cout<<L<<std::endl;
  for(int i=1;i!=n_aug_+1;i++)
  {
      Xsig_aug_.col(i) << (x_aug_ + (sqrt(lambda_+n_aug_)*L).col(i-1));
  }
  for(int i=n_aug_+1;i!=2*n_aug_+1;i++)
  {
      Xsig_aug_.col(i) << (x_aug_ - (sqrt(lambda_+n_aug_)*L).col(i-n_aug_-1));
  }

  Xsig_pred_.fill(0.0);

  for(int i=0;i!=2 * n_aug_ + 1;i++)
  {
      if(Xsig_aug_(4,i) == 0)
        Xsig_pred_(0,i) = Xsig_aug_(0,i) + Xsig_aug_(2,i)*cos(Xsig_aug_(3,i))*delta_t + 0.5*delta_t*delta_t*cos(Xsig_aug_(3,i))*Xsig_aug_(5,i);
      else
        Xsig_pred_(0,i) = Xsig_aug_(0,i) + Xsig_aug_(2,i)/Xsig_aug_(4,i)*(sin(Xsig_aug_(3,i)+Xsig_aug_(4,i)*delta_t) - sin(Xsig_aug_(3,i))) + 0.5*delta_t*delta_t*cos(Xsig_aug_(3,i))*Xsig_aug_(5,i);
      if(Xsig_aug_(4,i) == 0)
        Xsig_pred_(1,i) = Xsig_aug_(1,i) + Xsig_aug_(2,i)*sin(Xsig_aug_(3,i))*delta_t + 0.5*delta_t*delta_t*sin(Xsig_aug_(3,i))*Xsig_aug_(5,i);
      else
        Xsig_pred_(1,i) = Xsig_aug_(1,i) + Xsig_aug_(2,i)/Xsig_aug_(4,i)*(-cos(Xsig_aug_(3,i)+Xsig_aug_(4,i)*delta_t) + cos(Xsig_aug_(3,i))) + 0.5*delta_t*delta_t*sin(Xsig_aug_(3,i))*Xsig_aug_(5,i);
      Xsig_pred_(2,i) = Xsig_aug_(2,i) + delta_t*Xsig_aug_(5,i);
      Xsig_pred_(3,i) = Xsig_aug_(3,i) + Xsig_aug_(4,i)*delta_t + 0.5*delta_t*delta_t*Xsig_aug_(6,i);
      Xsig_pred_(4,i) = Xsig_aug_(4,i) + delta_t*Xsig_aug_(6,i);

  }

  weights_(0) = lambda_/(lambda_ + n_aug_);
  for(int i=1;i!=n_aug_*2+1;i++)
  {
      weights_(i) = 0.5/(lambda_ + n_aug_);
  }
  x_.fill(0.0);
  for(int i=0;i!=n_x_;i++)
  {
      double sum1 = 0;
      for(int j=0;j!=n_aug_*2+1;j++)
      {
          sum1 = sum1 + weights_(j)*Xsig_pred_(i,j);
      }
      x_(i) = sum1;
  }
  for(int i=0;i!=n_x_;i++)
  {
      for(int j=0;j!=n_x_;j++)
      {
          double sum1 = 0;
          for(int k=0;k!=n_aug_*2+1;k++)
          {
              sum1 = sum1 + weights_(k)*(Xsig_pred_(j,k) - x_(j))*(Xsig_pred_(i,k) - x_(i));
          }
          P_(i,j) = sum1;
      }
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  int n_z_ = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred_ = VectorXd(n_z_);

  //measurement covariance matrix S
  MatrixXd S_ = MatrixXd(n_z_,n_z_);

  for(int i=0;i!=2*n_aug_+1;i++)
  {
      Zsig_(0,i) = Xsig_pred_(0,i);
      Zsig_(1,i) = Xsig_pred_(1,i);
  }
  //calculate mean predicted measurement
  z_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++)
  {
      z_pred_ = z_pred_ + weights_(i)*Zsig_.col(i);
  }

  //calculate measurement covariance matrix S
  S_.fill(0.0);
  VectorXd z_diff_;
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_diff_ = Zsig_.col(i) - z_pred_;
    S_ = S_ + weights_(i) * z_diff_ * z_diff_.transpose();
  }

  MatrixXd R_ = MatrixXd(n_z_,n_z_);
  R_ <<   std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;
  S_ = S_ + R_;


  MatrixXd Tc_ = MatrixXd(n_x_, n_z_);
  Tc_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

    z_diff_ = Zsig_.col(i) - z_pred_;
    VectorXd x_diff_ = Xsig_pred_.col(i) - x_;
    Tc_ = Tc_ + weights_(i) * x_diff_ * z_diff_.transpose();
  }
  MatrixXd K_ = MatrixXd(n_x_, n_z_);
  VectorXd z_diff_NIS = meas_package.raw_measurements_ - z_pred_;
  MatrixXd S_i = S_.inverse();
  K_ = Tc_*S_i;
  x_ = x_ + K_*z_diff_NIS;
  P_ = P_ - K_*S_*K_.transpose();
  float NIS = z_diff_NIS.transpose()*S_i*z_diff_NIS;
  cout<<"LASER NIS: "<<NIS<<endl;
  //ofstream fout("NIS.txt",ios::app);
  //fout<<NIS<<endl;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  int n_z_ = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred_ = VectorXd(n_z_);

  //measurement covariance matrix S
  MatrixXd S_ = MatrixXd(n_z_,n_z_);

  for(int i=0;i!=2*n_aug_+1;i++)
  {
      Zsig_(0,i) = sqrt(Xsig_pred_(0,i)*Xsig_pred_(0,i) + Xsig_pred_(1,i)*Xsig_pred_(1,i));
      Zsig_(1,i) = atan2(Xsig_pred_(1,i), Xsig_pred_(0,i));
      Zsig_(2,i) = (Xsig_pred_(0,i)*cos(Xsig_pred_(3,i))*Xsig_pred_(2,i) + Xsig_pred_(1,i)*sin(Xsig_pred_(3,i))*Xsig_pred_(2,i))/Zsig_(0,i);
  }
  //calculate mean predicted measurement
  z_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++)
  {
      z_pred_ = z_pred_ + weights_(i)*Zsig_.col(i);
  }

  //calculate measurement covariance matrix S
  S_.fill(0.0);
  VectorXd z_diff_;
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_diff_ = Zsig_.col(i) - z_pred_;
    while (z_diff_(1)> M_PI)
      z_diff_(1)-=2.*M_PI;
    while (z_diff_(1)<-M_PI)
      z_diff_(1)+=2.*M_PI;
    S_ = S_ + weights_(i) * z_diff_ * z_diff_.transpose();
  }
  MatrixXd R_ = MatrixXd(n_z_,n_z_);
  R_ <<   std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S_ = S_ + R_;

  MatrixXd Tc_ = MatrixXd(n_x_, n_z_);
  Tc_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    //residual
    z_diff_ = Zsig_.col(i) - z_pred_;
    while (z_diff_(1)> M_PI)
      z_diff_(1)-=2.*M_PI;
    while (z_diff_(1)<-M_PI)
      z_diff_(1)+=2.*M_PI;

    VectorXd x_diff_ = Xsig_pred_.col(i) - x_;
    while (x_diff_(3)> M_PI)
      x_diff_(3)-=2.*M_PI;
    while (x_diff_(3)<-M_PI)
      x_diff_(3)+=2.*M_PI;

    Tc_ = Tc_ + weights_(i) * x_diff_ * z_diff_.transpose();
  }
  MatrixXd K_ = MatrixXd(n_x_, n_z_);
  VectorXd z_diff_NIS = meas_package.raw_measurements_ - z_pred_;
  MatrixXd S_i = S_.inverse();
  K_ = Tc_*S_i;
  x_ = x_ + K_*z_diff_NIS;
  P_ = P_ - K_*S_*K_.transpose();
  float NIS = z_diff_NIS.transpose()*S_i*z_diff_NIS;
  cout<<"RADAR NIS: "<<NIS<<endl;
  //ofstream fout("NIS.txt",ios::app);
  //fout<<NIS<<endl;

}
