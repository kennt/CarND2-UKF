#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.fill(0.0);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_.fill(0.0);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.7;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
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
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;

  // x_
  // P_
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;
  lambda_aug_ = 3 - n_aug_;

  // The weights are constant given lambda and n_aug
  // so create it here
  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(1/(2*(lambda_aug_ + n_aug_)));
  weights_(0) = lambda_aug_/(lambda_aug_ + n_aug_);
 
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  lidar_nis_stream_.open("lidar_nis.txt", ios::out | ios::trunc);
  radar_nis_stream_.open("radar_nis.txt", ios::out | ios::trunc);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_)
    return;
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_)
    return;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "UKF: " << endl;
    x_.fill(0.0);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      // raw measurements: ro, theta, rodot
      float r = meas_package.raw_measurements_(0);
      float theta = meas_package.raw_measurements_(1);

      x_(0) = r * cos(theta);
      x_(1) = r * sin(theta);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      // raw measurements: px, py
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
    }

    P_ <<   1,    0,    0,    0,    0,
            0,    1,    0,    0,    0,
            0,    0,    1,    0,    0,
            0,    0,    0,    1,    0,
            0,    0,    0,    0,    1;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  // convert to seconds
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}


VectorXd UKF::PredictAugVector(const VectorXd &inVector,
                               double delta_t)
{

  VectorXd result = VectorXd(n_x_);

  // Extract values from vector
  double v = inVector(2);
  double phi = inVector(3);
  double phi_dot = inVector(4);
  double noise_a = inVector(5);
  double noise_phi = inVector(6);
  
  // pre-calc some common values
  double cos_phi = cos(phi);
  double sin_phi = sin(phi);
  double dt2 = delta_t*delta_t / 2.0;

  // Avoid divide-by-zero
  //$ TODO: tolerance should be a parameter
  if (abs(phi_dot) < 0.001)
  {
      result <<
        v*cos_phi*delta_t + dt2*cos_phi*noise_a,
        v*sin_phi*delta_t + dt2*sin_phi*noise_a,
        0                 + delta_t*noise_a,
        0                 + dt2*noise_phi,
        0                 + delta_t*noise_phi;
  }
  else
  {
      result <<
        v*(sin(phi + phi_dot*delta_t) - sin_phi)/phi_dot + dt2*cos_phi*noise_a,
        v*(-cos(phi + phi_dot*delta_t) + cos_phi)/phi_dot + dt2*sin_phi*noise_a,
        0                               + delta_t*noise_a,
        phi_dot*delta_t                 + dt2*noise_phi,
        0                               + delta_t*noise_phi;
  }

  return result;
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);

  // This is similar to the lambda used, but this is adjusted
  // for the size of the augmented state
  double lambda_aug = 3 - n_aug_;

  //create augmented mean state
  x_aug.head(n_x_) = x_;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.block(0, 0, n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;
  
  //create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();
  
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  MatrixXd a_aug = sqrt(lambda_aug+n_aug_)*A_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) = a_aug.colwise() + x_aug;
  Xsig_aug.block(0, n_aug_+1, n_aug_, n_aug_) = (-a_aug).colwise() + x_aug;

  *Xsig_out = Xsig_aug;
}


void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
  Xsig.fill(0.0);

  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  //calculate sigma points ...
  //set sigma points as columns of matrix Xsig
  Xsig.col(0) = x_;
  MatrixXd a = sqrt(lambda_+n_x_)*A;
  Xsig.block(0, 1, n_x_, n_x_) = a.colwise() + x_;
  Xsig.block(0, n_x_+1, n_x_, n_x_) = (-a).colwise() + x_;

  //write result
  *Xsig_out = Xsig;
}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out,
                               const MatrixXd &Xsig_aug,
                               double delta_t) {
  MatrixXd Xsig_pred(n_x_, 2*n_aug_+1);
  Xsig_pred.fill(0.0);

  for (int i=0; i<Xsig_aug.cols(); i++)
  {
    Xsig_pred.col(i) = Xsig_aug.col(i).head(n_x_) + PredictAugVector(Xsig_aug.col(i), delta_t);
  }

  //write result
  *Xsig_out = Xsig_pred;
}

void NormalizeAngleInVector(VectorXd &v, int index)
{
  double angle = v(index);
  while (angle >  M_PI) angle-=2.*M_PI;
  while (angle < -M_PI) angle+=2.*M_PI;
  v(index) = angle; 
}


void UKF::PredictMeanAndCovariance(VectorXd* x_out,
                                   MatrixXd* P_out) {

  VectorXd  x = VectorXd(n_x_);
  MatrixXd  P = MatrixXd(n_x_, n_x_);

  //predict state mean
  x.fill(0.0);
  P.fill(0.0);
  
  for (int i=0; i<Xsig_pred_.cols(); i++)
  {
      x += weights_(i) * Xsig_pred_.col(i);
  }
  
  //predict state covariance matrix
  for (int i=0; i<Xsig_pred_.cols(); i++)
  {
      VectorXd temp = Xsig_pred_.col(i) - x;
      NormalizeAngleInVector(temp, 3);
      P += weights_(i) * temp * temp.transpose();
  }
  *x_out = x;
  *P_out = P;
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  GenerateSigmaPoints(&Xsig_pred_);

  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_ + 1);
  AugmentedSigmaPoints(&Xsig_aug);

  SigmaPointPrediction(&Xsig_pred_, Xsig_aug, delta_t);

  PredictMeanAndCovariance(&x_, &P_);
}

void NormalizeAngleInMatrix(MatrixXd &m, int index)
{
  for (int i=0; i<m.cols(); i++)
  {
    double angle = m(index, i);
    while (angle> M_PI) angle-=2.*M_PI;
    while (angle<-M_PI) angle+=2.*M_PI;
    m(index, i) = angle;
  }
}


void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
    
  //transform sigma points into measurement space
  for (int i=0; i<Zsig.cols(); i++)
  {
      float px = Xsig_pred_(0, i);
      float py = Xsig_pred_(1, i);
      float p2 = px*px + py*py;
      float v = Xsig_pred_(2, i);
      float phi = Xsig_pred_(3, i);
      
      if (abs(p2) < 0.0001)
        continue;
        
      float rho = sqrt(p2);
      Zsig.col(i) << rho,
                     atan2(py, px),
                     (px*cos(phi)*v + py*sin(phi)*v)/rho;
  }
  
  //calculate mean predicted measurement
  for (int i=0; i<Zsig.cols(); i++)
  {
      z_pred += weights_(i) * Zsig.col(i);
  }

  //calculate innovation covariance matrix S
  // Fill S with the values of R
  S.fill(0.0);
  S(0,0) = std_radr_ * std_radr_;
  S(1,1) = std_radphi_ * std_radphi_;
  S(2,2) = std_radrd_ * std_radrd_;
  for (int i=0; i<Zsig.cols(); i++)
  {
      VectorXd temp = Zsig.col(i) - z_pred;
      NormalizeAngleInVector(temp, 1);
      S += weights_(i) * temp * temp.transpose();
  }

  // Go through and normalize the angles
  //NormalizeAngleInMatrix(S, 1);

  //write result
  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;
}

void UKF::UpdateRadarState(VectorXd* x_out,
                           MatrixXd* P_out,
                           const VectorXd &z,
                           const VectorXd &z_pred,
                           const MatrixXd &Zsig,
                           const MatrixXd &S) {

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = z.size();

  VectorXd x = x_;
  MatrixXd P = P_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i=0; i<Xsig_pred_.cols(); i++)
  {
      VectorXd zdiff = Zsig.col(i) - z_pred;
      NormalizeAngleInVector(zdiff, 1);

      VectorXd xdiff = Xsig_pred_.col(i) - x_;
      NormalizeAngleInVector(xdiff, 3);

      Tc += weights_(i) * xdiff * zdiff.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  VectorXd zdiff = z - z_pred;

  x += K * zdiff;
  P -= K * S * K.transpose();

  //write result
  *x_out = x;
  *P_out = P;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = 3;
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_.head(n_z);

  VectorXd z_pred = VectorXd(n_z);
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  PredictRadarMeasurement(&z_pred, &S, &Zsig);
  UpdateRadarState(&x_, &P_, z, z_pred, Zsig, S);

  // Calculate NIS and write out to a file
  VectorXd zdiff = z - z_pred;
  double nis = zdiff.transpose() * S.inverse() * zdiff;
  radar_nis_stream_ << time_us_ << " : " << nis << endl;
}

void UKF::PredictLidarMeasurement(VectorXd* z_out, MatrixXd* S_out, MatrixXd* Zsig_out) {

  //set measurement dimension, lidar can measure px, py
  int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for (int i=0; i<Zsig.cols(); i++)
  {
      float px = Xsig_pred_(0, i);
      float py = Xsig_pred_(1, i);
      
      Zsig.col(i) << px, py;
  }
  
  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i<Zsig.cols(); i++)
  {
      z_pred += weights_(i) * Zsig.col(i);
  }

  //calculate innovation covariance matrix S
  // Fill S with the values of R
  S.fill(0.0);
  S(0,0) = std_laspx_ * std_laspx_;
  S(1,1) = std_laspy_ * std_laspy_;
  for (int i=0; i<Zsig.cols(); i++)
  {
      VectorXd temp = Zsig.col(i) - z_pred;
      S += weights_(i) * temp * temp.transpose();
  }

  //write result
  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  int n_z = 2;
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_.head(n_z);

  VectorXd z_pred = VectorXd(n_z);
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  PredictLidarMeasurement(&z_pred, &S, &Zsig);
  UpdateRadarState(&x_, &P_, z, z_pred, Zsig, S);

  VectorXd zdiff = z - z_pred;
  double nis = zdiff.transpose() * S.inverse() * zdiff;
  lidar_nis_stream_ << time_us_ << " : " << nis << endl;
}


