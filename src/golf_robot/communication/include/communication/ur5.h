/*-------------------------------------------------------------------------------------------------------------------------------------------------------
 * Fault-tolerant Control of Field Robot                                                                                                              F08
 *                                                                                                                                  S�ren Hansen, s021751
 * linreg.h:  Header file for the linear regression function. Defines the data type used.                                           Peter Tjell,  s032041
 *-------------------------------------------------------------------------------------------------------------------------------------------------------
 */
 
# ifndef UR5_H
# define UR5_H



// Defines functions as external C code. For use in C++ programs.
#ifdef __cplusplus
 extern "C" {
#endif
typedef struct{
		double R[9];
		double O[3];
}tfrotype;

  void ujac(double *J,double * v,double * apar,double * dpar);  
   void ufwdkin(tfrotype  *,double * v,double * apar,double * dpar);  
   

#ifdef __cplusplus
 }
#endif

# endif // UR5_H
