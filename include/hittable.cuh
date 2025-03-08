#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "ray.cuh"
#include "vec3.cuh"

class HitRecord {
public:
  Point3 p;
  Vec3 normal;
  double t;
  bool front_face;
  DEVICE void set_front_face(const Ray &r, const Vec3 &outward_normal) {
    front_face = (dot(r.direction(), outward_normal) < 0);
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class Hittable {
public:
  DEVICE virtual ~Hittable() {};
  DEVICE virtual bool hit(const Ray &r, double t_min, double t_max,
                          HitRecord &rec) const = 0;
};

#endif
