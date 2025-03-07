#ifndef VEC3_CUH
#define VEC3_CUH
#include "utils.cuh"
#include <cmath>
#include <cstdio>
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

  HOST_DEVICE Vec3 &operator-=(const Vec3 &other) {
    *this += -other;
    return *this;
  }

  HOST_DEVICE Vec3 &operator*=(const Vec3 &other) {
    _x *= other.x();
    _y *= other.y();
    _z *= other.z();
    return *this;
  }

  HOST_DEVICE Vec3 &operator*=(const double &scalar) {
    _x *= scalar;
    _y *= scalar;
    _z *= scalar;
    return *this;
  }

  HOST_DEVICE Vec3 &operator/=(const Vec3 &other) {
    _x /= other.x();
    _y /= other.y();
    _z /= other.z();
    return *this;
  }

  HOST_DEVICE Vec3 &operator/=(const double &scalar) {
    *this *= 1 / scalar;
    return *this;
  }

  HOST_DEVICE double length_squared() const {
    return _x * _x + _y * _y + _z * _z;
  }

  HOST_DEVICE double length() const { return std::sqrt(length_squared()); }

private:
  double _x, _y, _z;
};

HOST_DEVICE inline void print_vec3(const Vec3 &v) {
  printf("(%f, %f, %f)\n", v.x(), v.y(), v.z());
}

HOST_DEVICE inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z());
}

HOST_DEVICE inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2) {
  return v1 + (-v2);
}

HOST_DEVICE inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z());
}

HOST_DEVICE inline Vec3 operator*(const Vec3 &v, const double &scalar) {
  return Vec3(v.x() * scalar, v.y() * scalar, v.z() * scalar);
}

HOST_DEVICE inline Vec3 operator*(const double &scalar, const Vec3 &v) {
  return v * scalar;
}

HOST_DEVICE inline Vec3 operator/(const Vec3 &v, const double &scalar) {
  return v * (1 / scalar);
}

HOST_DEVICE inline double dot(const Vec3 &v1, const Vec3 &v2) {
  return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

HOST_DEVICE inline Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.y() * v2.z() - v1.z() * v2.y(),
              v1.z() * v2.x() - v1.x() * v2.z(),
              v1.x() * v2.y() - v1.y() * v2.x());
}

using Point3 = Vec3;

#endif
