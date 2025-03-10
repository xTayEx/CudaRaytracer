#ifndef SPHERE_CUH
#define SPHERE_CUH

#include "hittable.cuh"
#include "material.cuh"

class Sphere : public Hittable {
public:
  DEVICE Sphere(Point3 center, double radius, Material *mat)
      : center(center), radius(radius), mat(mat) {}
  DEVICE ~Sphere(){};
  DEVICE bool
  hit(const Ray &r, double t_min, double t_max, HitRecord &rec) const override {

    Vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = h * h - a * c;

    if (discriminant < 0) {
      return false;
    }

    auto sqrtd = std::sqrt(discriminant);
    auto root = (h - sqrtd) / a;
    if (root <= t_min || root >= t_max) {
      root = (h + sqrtd) / a;
      if (root <= t_min || root >= t_max) {
        return false;
      }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    auto outward_normal = (rec.p - center) / radius;
    rec.set_front_face(r, outward_normal);
    rec.mat = mat;

    return true;
  }

private:
  Point3 center;
  double radius;
  Material *mat;
};

#endif
