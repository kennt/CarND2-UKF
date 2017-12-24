#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;


Tools::Tools(int state_vector_length)
  : state_vector_length_(state_vector_length)
{}

Tools::~Tools() {}

//
// Calculates the RMSE (Root Mean-Squared Error) of the
// estimates from the actual value.
//
// If the length of the estimates is 0 or if number of
// estimates does not match up the actual values, then
// a vector of all zeros is returned and an error message
// is written to stdout (cout).
//
VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd  rmse(state_vector_length_);

  // Initialize to all zeros
  rmse.fill(0.0);

  // Sanity checks
  if (estimations.size() == 0)
  {
    cout << "Tools::CalculateRMSE " << __LINE__ << " : Error : "
         << " estimations vector is length 0"
         << std::endl;
    return rmse;
  }

  // Are the vector of predictions and actual values the same?
  if (estimations.size() != ground_truth.size())
  {
    cout << "Tools::CalculateRMSE " << __LINE__ << " : Error : "
         << "estimations and ground_truth vector have different lengths"
         << std::endl;
    return rmse;
  }

  // summ up all the residuals
  for (auto i=0; i<estimations.size(); i++)
  {
    VectorXd residual = estimations[i] - ground_truth[i];
    rmse.array() += residual.array().pow(2);
  }

  // calculate the mean
  rmse /= estimations.size();

  // take the square root
  return rmse.array().sqrt();
}
