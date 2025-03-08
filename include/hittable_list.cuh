#ifndef HITTABLE_LIST_CUH
#define HITTABLE_LIST_CUH

#include "hittable.cuh"
#include <cstdlib>
#include <thrust/device_vector.h>

class HittableList : public Hittable {
public:
  DEVICE HittableList() {}
  DEVICE HittableList(int capacity) {
    __shared__ Hittable **shared_objects;
    __shared__ int shared_capacity;

    if (threadIdx.x == 0) {
      shared_capacity = capacity;
      shared_objects = (Hittable **)malloc(capacity * sizeof(Hittable *));
    }
    __syncthreads();

    objects = shared_objects;
    capacity = shared_capacity;
    size = 0;
  }

  DEVICE void clear() { size = 0; }

  DEVICE void add(Hittable *object) {
    int old = atomicAdd(&size, 1);
    if (old < capacity) {
      objects[old] = object;
    }
  }

  DEVICE bool hit(const Ray &r, double t_min, double t_max,
                  HitRecord &rec) const override {
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < size; ++i) {
      auto current_object = objects[i];
      if (current_object->hit(r, t_min, closest_so_far, temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }

    return hit_anything;
  }

  DEVICE ~HittableList() {
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i = 0; i < size; ++i) {
        // TODO: check if this way of freeing is correct or not.
        free(objects[i]);
      }
      free(objects);
    }
  };

private:
  Hittable **objects;
  int capacity;
  int size;
};

#endif
