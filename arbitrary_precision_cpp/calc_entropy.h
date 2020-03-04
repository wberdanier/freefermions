#undef EIGEN_NO_DEBUG
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "Eigen/Eigen"
#include "Eigen/Eigenvalues"
#include <cmath>
#include "mpreal.h"
#include "MPRealSupport.h"
using namespace std;
using mpfr::mpreal;

typedef Eigen::Matrix<mpreal,Eigen::Dynamic,Eigen::Dynamic> MatrixDD; // long doubles - change in calc_entropy.cxx too
// typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixDD; // regular doulbes
typedef Eigen::Matrix<complex<mpreal>,Eigen::Dynamic,Eigen::Dynamic> MatrixCD;

typedef Eigen::Matrix<complex<mpreal> ,Eigen::Dynamic,1> VectorCD;

void print_matrix(const MatrixDD & m);
// void print_vector(const vector<double> & v);
void construct_F(const vector<double> & phi1, const vector<double> & phi2, MatrixDD & F);
void diag_F(const MatrixDD & F, MatrixDD & Q, MatrixDD & F_bd);
double max_val(const MatrixDD & m);
double entropy(vector<double> phi1,vector<double> phi2, vector<double> flips, int digits);
void get_corr_matrix(const vector<double> & flips,const MatrixDD & Q,MatrixDD & C);
