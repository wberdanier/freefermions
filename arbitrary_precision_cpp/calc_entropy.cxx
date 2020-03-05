#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "Eigen/Eigen"
#include "Eigen/Eigenvalues"
#include <cmath>
#include "calc_entropy.h"
#include "mpreal.h"
#include "MPRealSupport.h"
//#include <boost/multiprecision/cpp_dec_float.hpp>
using namespace std;
using mpfr::mpreal;

////////////////////////////////////////////////
// This code calculate the entanglement entropy of a disordered driven 1D quantum system of free Majoranas.
// It uses the Eigen package, as well as MPFR for arbitrary precision numerics.
// See PNAS 115 (38) 9491-9496 (2018) for physics details.
// Also included in this directory is python code for calling these C++ methods using the SWIG interface.
////////////////////////////////////////////////

// typedef long double mydtype; // long doubles
// typedef double mydtype; // regular doubles

typedef Eigen::Matrix<mpreal,Eigen::Dynamic,Eigen::Dynamic> MatrixDD;
//typedef boost::multiprecision::cpp_dec_float_50 mydtype;
//typedef Eigen::Matrix<mydtype,Eigen::Dynamic,Eigen::Dynamic> MatrixDD;

typedef Eigen::Matrix<complex<mpreal>,Eigen::Dynamic,Eigen::Dynamic> MatrixCD;

typedef Eigen::Matrix<complex<mpreal> ,Eigen::Dynamic,1> VectorCD;

// namespace Eigen {
//   template<> struct NumTraits<mydtype>
//   {
//     typedef mydtype Real;
//     typedef mydtype NonInteger;
//     enum {
//       IsComplex = 0,
//       IsInteger = 0,
//       IsSigned = 1,
//       RequireInitialization = 1,
//       ReadCost = 1, // See https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html
//       AddCost = 3,
//       MulCost = 3
//     };
//   };
// };

void get_corr_matrix(const vector<double> & flips,const MatrixDD & Q,MatrixDD & C);

double entropy(vector<double> phi1,vector<double> phi2, vector<double> flips, int digits) {
  using mpfr::mpreal;
  // Required precision of computations in decimal digits.
  // Play with it to check different precisions.
  // const int digits = 50;
  // Setup default precision for all subsequent computations
  // MPFR accepts precision in bits - so we do the conversion
  mpreal::set_default_prec(mpfr::digits2bits(digits));


  int L=phi1.size();
  assert(int(phi2.size())==L);
  int N=2*L;


  MatrixDD F=MatrixDD::Zero(N,N);
  MatrixDD Q=MatrixDD::Zero(N,N);
  MatrixDD F_bd=MatrixDD::Zero(N,N);

  // Construct the F matrix with mydtype precision in Eigen3
  construct_F(phi1,phi2,F);

  // Block diagonalize F
  diag_F(F,Q,F_bd);

  // Normalize the eigenvectors
  for(int col=0;col<Q.cols();col+=1) {
    Q.col(col)/=Q.col(col).norm();
  }

  // // Write matrices to file
  // // save F
  // ofstream out1("printmats/F.out");
  // out1.precision(15);
  // for(int j = 0; j < F.rows(); j++){
  //   for(int k = 0; k < F.cols(); k++){
  //     out1 << F(j,k) << ' ';
  //   }
  //   out1 << endl;
  // }

  // ofstream out2("printmats/Q.out");
  // out2.precision(15);
  // for(int j = 0; j < Q.rows(); j++){
  //   for(int k = 0; k < Q.cols(); k++){
  //     out2 << Q(j,k) << ' ';
  //   }
  //   out2 << endl;
  // }


  // // Print matrices
  // cerr << "F=" << endl;
  // print_matrix(F);
  // cerr << endl;
  //
  // cerr << "F*F^T=" << endl;
  // print_matrix(F*F.transpose());
  // cerr << endl;
  //
  // cerr << "Q=" << endl;
  // print_matrix(Q);
  // cerr << endl;
  //
  // cerr << "F_bd=" << endl;
  // print_matrix(F_bd);
  // cerr << endl;
  //
  // // Check that it is block diagonalized appropriately
  // MatrixDD test=Q*F_bd*Q.transpose();
  // cerr << "Max value of F - Q*F_bd*QT is " << max_val(F-test) << endl;
  //
  // test=F_bd;
  // for(int j=0;j<test.rows();j+=2)
  //   test(j,j)=test(j,j+1)=test(j+1,j)=test(j+1,j+1)=0;
  // cerr << "Max value of off-diagonals in F_bd is " << max_val(test) << endl;
  //
  // test=Q*Q.transpose();
  // for(int j=0;j<test.rows();j++)
  //   test(j,j)-=1.;
  // cerr << "Max value of Q*Q^T - 1 is " << max_val(test) << endl;
  //
  // test=Q.transpose()*Q;
  // for(int j=0;j<test.rows();j++)
  //   test(j,j)-=1.;
  // cerr << "Max value of Q^T*Q - 1 is " << max_val(test) << endl;

  // Calculate the correlation matrix for a cut at L/2
  MatrixDD C = MatrixDD::Zero(L,L); // L is half the system, 2L size matrices
  get_corr_matrix(flips,Q,C);

  // Calculate the entropy from the correlation matrix
  complex<mpreal> I = complex<mpreal>(0,1); // define i
  Eigen::SelfAdjointEigenSolver<MatrixCD> es(I * C);
  MatrixCD lam_C = es.eigenvalues();

  // VectorCD lam_C = es.eigenvalues();
  double S = 0.;
  double p;
  double mu;
  for(int z=0; z<C.rows(); z++){
    mu = abs(lam_C(z)).toDouble();
    p = (1. - mu) / 2.;
    if(p>0. && p<1.){
      S -=  ( p * log(p) + (1. - p) * log(1. - p) ); // C has eigenvalues at +lambda and -lambda; thus mult by 2 and only sum positive ones
    }
  }
  S /= 2;

  // // Save lam_C matrix
  // ofstream out10("printmats/lam_C.out");
  // out10.precision(15);
  // for(int j = 0; j < lam_C.rows(); j++){
  //   for(int k = 0; k < lam_C.cols(); k++){
  //     out10 << lam_C(j,k) << ' ';
  //   }
  //   out10 << endl;
  // }

  // // Print stuff
  // cerr << "C=" << endl;
  // print_matrix(C);
  // cerr << endl;
  //
  // cerr << "S= " << S << endl;

  // // Save C matrix
  // ofstream out3("printmats/C.out");
  // out3.precision(15);
  // for(int j = 0; j < C.rows(); j++){
  //   for(int k = 0; k < C.cols(); k++){
  //     out3 << C(j,k) << ' ';
  //   }
  //   out3 << endl;
  // }

  return S;
}

void get_corr_matrix(const vector<double> & flips,const MatrixDD & Q,MatrixDD & C){
  // Construct C in block-diagonal form; +1 is on, -1 is off for each mode
  MatrixDD Cslice = MatrixDD::Zero(Q.rows(),Q.rows());

  for(int j=0; j<Q.rows(); j+=2){
    if(flips.at(j/2) > 0.5){

      // // // Print flips list
      // cerr << "flips: " << flips.at(j/2) << endl;

      Cslice(j,j+1) = 1.;
    }else{
      Cslice(j,j+1) = -1.;
    }
    Cslice(j+1,j) = -Cslice(j,j+1);
  }

  // // save Cslice before rotating back
  // ofstream out("printmats/Cslice.out");
  // out.precision(15);
  // for(int j = 0; j < Cslice.rows(); j++){
  //   for(int k = 0; k < Cslice.cols(); k++){
  //     out << Cslice(j,k) << ' ';
  //   }
  //   out << endl;
  // }

  // Rotate back from the diagonal basis
  Cslice = Q * Cslice * Q.transpose();

  // // // Print Cslice
  // cerr << "Cslice=" << endl;
  // print_matrix(Cslice);
  // cerr << endl;

  // Slice at l = L/2
  for(int k=0; k<Cslice.rows()/2; k++){
    for(int l=0; l<Cslice.rows()/2; l++){
      C(k,l) = Cslice(k,l);
    }
  }
}

void construct_F(const vector<double> & phi1, const vector<double> & phi2, MatrixDD & F) {
  MatrixDD F1 = F, F2 = F;
  int L = F.rows();
  for(int j=0; j < L; j+=2){
    F1(j,j) = cos(phi1.at(j/2));
    F1(j+1,j+1) = cos(phi1.at(j/2));
    F1(j,j+1) = sin(phi1.at(j/2));
    F1(j+1,j) = -sin(phi1.at(j/2));
  }
  for(int j=1; j < L - 1; j+=2){
    F2(j,j) = cos(phi2.at((j-1)/2));
    F2(j+1,j+1) = cos(phi2.at((j-1)/2));
    F2(j,j+1) = sin(phi2.at((j-1)/2));
    F2(j+1,j) = -sin(phi2.at((j-1)/2));
  }

  // PBCs
  double phiL = phi2.at(L/2-1);
  F2(0,0) = cos(phiL);
  F2(0,L-1) = -sin(phiL);
  F2(L-1,0) = sin(phiL);
  F2(L-1,L-1) = cos(phiL);

  // Print some stuff
  // cerr << "phiL" << endl;
  // cerr << phiL << endl;
  // cerr << endl;
  //
  // cerr << "L" << endl;
  // cerr << L << endl;
  //
  // cerr << "phi2" << endl;
  // for(int j=0; j<phi2.size(); j++){
  //   cerr << phi2[j] << ' ';
  // }
  // cerr << endl;


  F = F2 * F1;

// // Print matrices
//   cerr << "F1=" << endl;
//   print_matrix(F1);
//   cerr << endl;
//
//   cerr << "F2=" << endl;
//   print_matrix(F2);
//   cerr << endl;
//
//   cerr << "F=" << endl;
//   print_matrix(F);
//   cerr << endl;


}

void diag_F(const MatrixDD & F, MatrixDD & Q, MatrixDD & F_bd) {
  Eigen::EigenSolver<MatrixDD> es(F);
  F_bd = es.pseudoEigenvalueMatrix();
  Q = es.pseudoEigenvectors();
}

void print_matrix(const MatrixDD & m) {
  for(int j=0; j<m.rows(); j++) {
    for(int k=0; k<m.cols(); k++) {
      cerr << m(j,k) << ' ';
    }
    cerr << endl;
  }
}

// void print_vector(const vector<double> & v){
//   for(int j=0; j<v.size(); j++){
//     cerr << v[j] << ' ';
//   }
//   cerr << endl;
// }

double max_val(const MatrixDD & m) {
  mpreal max_diff=0.;
  for(int j=0;j< m.rows();j++)
    for(int k=0;k<m.cols();k++)
      max_diff=max(max_diff,abs(m(j,k)));
  return max_diff.toDouble();
}

int main(int argc,char ** argv){
  ifstream in("f.in");
  int L;
  vector<double> phi1, phi2;
  vector<double> flips;

  in >> L;
  double temp;
  bool tempbool;

  for(int x=0; x<L; x++){
    in >> temp;
    phi1.push_back(temp);
  }
  for(int y=0; y<L; y++){
    in >> temp;
    phi2.push_back(temp);
  }

  for(int i=0; i<L; i++){
    in >> tempbool;
    flips.push_back(double(tempbool));
  }


  double entropy_calculated = entropy(phi1,phi2,flips,50);

  ofstream out("f.out");
  out.precision(15);
  out << entropy_calculated << endl;

  return 0;
}
