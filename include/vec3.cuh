#ifndef VEC3_CUH
#define VEC3_CUH
#include "utils.cuh"
#include <cuda_runtime.h>

class Vec3 {
public:
  HOST_DEVICE Vec3() : _x(0.0f), _y(0.0f), _z(0.0f) {}
  HOST_DEVICE Vec3(double x_val, double y_val, double z_val)
      : _x(x_val), _y(y_val), _z(z_val) {}

  HOST_DEVICE double x() const { return _x; }
  HOST_DEVICE double y() const { return _y; }
  HOST_DEVICE double z() const { return _z; }

  HOST_DEVICE Vec3 operator-() const { return Vec3(-_x, -_y, -_z); }

  HOST_DEVICE double operator[](int i) const {
    if (i == 0) {
      return _x;
    } else if (i == 1) {
      return _y;
    } else {
      return _z;
    }
  }

  HOST_DEVICE double &operator[](int i) {
    if (i == 0) {
      return _x;
    } else if (i == 1) {
      return _y;
    } else {
      return _z;
    }
  }

  HOST_DEVICE Vec3 &operator+=(const Vec3 &other) {
    _x += other.x();
    _y += other.y();
    _z += other.z();
    return *this;
  }

private:
  double _x, _y, _z;
};

#endif
