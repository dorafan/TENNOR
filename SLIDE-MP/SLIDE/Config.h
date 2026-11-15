#define ADAM 1
#define BETA1 0.9
#define BETA2 0.999
#define EPS 0.00000001

//1: wta; 2: Densified wta; 3: topk minhash; 4: simhash
#define HashFunction 2
extern int BUCKETSIZE;

//for minhash
#define TOPK 30
//for simhash
#define Ratio 3

//for wta/dwta
extern int binsize;

extern int isBruteForce;
extern int NNSorMPS;
extern int BFnumber;

extern int isMPWTA;

extern int isRandom;

extern int isHopper;

extern int isLog;

//Mode 1: Topk thresholding Mode 4: Sampling
#define Mode 4

#define THRESH 2

#define FIFO 1

extern int LOADWEIGHT;

#define MAPLEN 325056

#define MP 1

#include <string>

extern std::string retrivalfile;  
