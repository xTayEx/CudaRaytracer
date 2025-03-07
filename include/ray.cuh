#ifndef RAY_CUH
#define RAY_CUH

#include "utils.cuh"
#include "vec3.cuh"

class Ray {
public:
  HOST_DEVICE Ray() {}
  HOST_DEVICE Ray(const Vec3 &origin, const Vec3 &direction)
      : _origin(origin), _direction(direction) {}

  HOST_DEVICE Vec3 origin() const { return _origin; }
  HOST_DEVICE Vec3 direction() const { return _direction; }

  HOST_DEVICE Vec3 at(double t) const { return _origin + t * _direction; }

private:
  Vec3 _origin;
  Vec3 _direction;
};

#endif
