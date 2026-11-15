#define ADAM 1
#define BETA1 0.9
#define BETA2 0.999
#define EPS 0.00000001

//1: wta; 2: Densified wta; 3: topk minhash; 4: simhash
#define HashFunction 2
#define BUCKETSIZE 128

#define TOPK 30

#define Ratio 3

#define binsize 8
//Mode 1: Topk thresholding Mode 4: Sampling
#define Mode 4

#define THRESH 2

#define FIFO 1

#define LOADWEIGHT 0

#define MAPLEN 325056

#define SERIAL_LEN 6144  // 5677 bytes ~= 44 kb -> 48kb = 6144 Bytes

// build.sh --dataset supplies ORAM_N through CMake; otherwise use this default.
#ifndef ORAM_N
#define ORAM_N 65536 // 30938 + 512*50 = 56538 -> 65536 = 2^16
#endif

#define TEST 1

extern int ORAMtime;
extern int stashtime;
extern int positionmaptime;
extern int serializetime;

extern int oraminittime;

extern int oramfirstwritetime;

extern int updatetime;
extern int rehashtime;
extern int updatenodetime;

extern int updateNodeOramTime;

extern int rehashLshOramTime;

extern int feedforwardtime;
extern int backpropagationtime;
extern int lshquerytime;

extern int oramnodetime;

extern int oramlshtime;

