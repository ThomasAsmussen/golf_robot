
#include <iostream>
#include <string.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <time.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#include <gsl/gsl_math.h>

#include "../include/communication/ur_driver.h"
#include "../include/communication/ur5.h"


void rotate(gsl_vector *res,gsl_matrix *R, gsl_vector *inp,gsl_vector *t1,gsl_vector *t2){
      t1->data=inp->data;
      t2->data=res->data;
      gsl_blas_dgemv(CblasNoTrans ,1.0,R, t1,0.0,t2); 
      t1->data=&inp->data[3];
      t2->data=&res->data[3];
      gsl_blas_dgemv(CblasNoTrans ,1.0,R, t1,0.0,t2); 
}  
  
int main()
{
  std::condition_variable rt_msg_cond_;
  std::condition_variable msg_cond_;

//  double apar[]={0, -0.42500, -0.39243,0,0,0} ;
 // double dpar[]={ 0.0892,0,0, 0.10900,  0.09300,  0.0820};
  double apar[]={0, -0.6120, -0.5723,0,0,0} ;
  double dpar[]={ 0.1273,0,0, 0.163941 ,  0.1157,  0.0922};
  tfrotype tfkin;
  double v[6];
  double r2d=180/3.1415;
  UrDriver mydriver(rt_msg_cond_, msg_cond_,"192.38.66.227",5007);
  mydriver.start();
  std:: vector<double> q(6);
  std:: vector<double> f(6);
  gsl_vector *x=gsl_vector_alloc(6);
  gsl_matrix *A=gsl_matrix_alloc(6,6);
  gsl_matrix *R=gsl_matrix_alloc(3,3);
  gsl_permutation *p=gsl_permutation_alloc(6);
  gsl_vector * vw_w=gsl_vector_alloc(6);
  gsl_vector * vw_t=gsl_vector_alloc(6);
  gsl_vector * t1=gsl_vector_alloc(3);
  gsl_vector * t2=gsl_vector_alloc(3);
  int signum,i,j;
  R->data=tfkin.R;
  std::cout << "I am started \n";
  std::mutex msg_lock; // The values are locked for reading in the class, so just use a dummy mutex
  std::unique_lock<std::mutex> locker(msg_lock);
			while (!mydriver.rt_interface_->robot_state_->getDataPublished()) {
				rt_msg_cond_.wait(locker);
			}
  q= mydriver.rt_interface_->robot_state_->getQActual();
  std::cout << q.data()[0]*r2d<<" " << q.data()[1]*r2d <<" "<< q.data()[2]*r2d<<" " << q.data()[3]*r2d<<" " << q.data()[4]*r2d<<" " << q.data()[5]*r2d << "\n";
 
  mydriver.rt_interface_->robot_state_->setDataPublished();
  ufwdkin(&tfkin,q.data(),apar,dpar);
  mydriver.setSpeed(0.0,0,0,0,0,0.0,1,0.008);
  gsl_vector_set(vw_t,3,0.00);gsl_vector_set(vw_t,4,0.0);gsl_vector_set(vw_t,5,0.0);
  for (int mc=0;mc <1;mc++){
     switch (mc) {
       case 0: gsl_vector_set(vw_t,0,0.00);gsl_vector_set(vw_t,1,0.0);gsl_vector_set(vw_t,2,0);
       break;
      
       case 1: gsl_vector_set(vw_t,0,0.0);gsl_vector_set(vw_t,1,0.05);gsl_vector_set(vw_t,2,0);
       break;
       
       case 2: gsl_vector_set(vw_t,0,-0.05);gsl_vector_set(vw_t,1,0.0);gsl_vector_set(vw_t,2,0);
       break;
       
       case 3: gsl_vector_set(vw_t,0,0.0);gsl_vector_set(vw_t,1,-0.05);gsl_vector_set(vw_t,2,0);
       break;
       default:;
     }
     rotate(vw_w,R,vw_t,t1,t2);
    
   for (int i=0;i<125*10;i++){
    // Wait for next sample 
    std::mutex msg_lock; // The values are locked for reading in the class, so just use a dummy mutex
    std::unique_lock<std::mutex> locker(msg_lock);
    while (!mydriver.rt_interface_->robot_state_->getDataPublished()) {
				rt_msg_cond_.wait(locker);
    }
    
    q= mydriver.rt_interface_->robot_state_->getQActual();
    f= mydriver.rt_interface_->robot_state_->getTcpForce();
    mydriver.rt_interface_->robot_state_->setDataPublished();
    gsl_vector_set(vw_t,1,0.2*sin(i/125.0*2*M_PI));
    ufwdkin(&tfkin,q.data(),apar,dpar); 
    rotate(vw_w,R,vw_t,t1,t2);
    // std::cout << q.data()[0]*r2d<<" " << q.data()[1]*r2d <<" "<< q.data()[2]*r2d<<" " << q.data()[3]*r2d<<" " << q.data()[4]*r2d<<" " << q.data()[5]*r2d << "\n";
    //calculate jacobian and find the joint velocities 
    ujac(A->data,q.data(),apar,dpar); 
    gsl_linalg_LU_decomp(A,p,&signum);
    gsl_linalg_LU_solve(A,p,vw_w,x);
    for (int k=0;k<6;k++){
      v[k]=gsl_vector_get(x,k);
      if (v[k] > 1) v[k]=1;
      if (v[k] < -1) v[k]=-1;
    }   
    mydriver.setSpeed(v[0],v[1],v[2],v[3],v[4],v[5],10,0.008);
    if (i % 62 == 0){
       // std::cout << v[0]<<" " << v[1]<<" "<< v[2]<<" " <<v[3]<<" " << v[4]<<" " << v[5] << "\n";
	std::cout << f[0]<<" " << f[1]<<" "<< f[2]<<" " <<f[3]<<" " << f[4]<<" " << f[5] << "\n";
    }
  }
}
  mydriver.setSpeed(0.0,0,0,0,0,0.0,1,0.008);
  mydriver.halt();
  std::cout << "Program ended!\n";
}
