#ifndef UTILS_CUH
#define UTILS_CUH

#define HOST_DEVICE __host__ __device__
#define HOST __host__
#define DEVICE __device__

HOST_DEVICE double clamp(double val, double mini, double maxi) {
  return val < mini ? mini : (val > maxi ? maxi : val);
}

#endif
