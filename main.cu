#include "camera.cuh"
#include "hittable_list.cuh"
#include "material.cuh"
#include "random.cuh"
#include "sphere.cuh"
#include "utils.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

__global__ void initialize_camera(Camera *camera_ptr,
                                  double lookfrom_x,
                                  double lookfrom_y,
                                  double lookfrom_z,
                                  double lookat_x,
                                  double lookat_y,
                                  double lookat_z,
                                  double vup_x,
                                  double vup_y,
                                  double vup_z,
                                  int image_width,
                                  int image_height,
                                  double samples_per_pixel,
                                  double vfov,
                                  double defocus_angle,
                                  double focus_dist) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (camera_ptr) Camera();
    camera_ptr->intialize(Point3(lookfrom_x, lookfrom_y, lookfrom_z),
                          Point3(lookat_x, lookat_y, lookat_z),
                          Vec3(vup_x, vup_y, vup_z),
                          image_width,
                          image_height,
                          samples_per_pixel,
                          vfov,
                          defocus_angle,
                          focus_dist);
  }
}

__global__ void initialize_world(HittableList *world_ptr,
                                 Sphere **spheres_ptr,
                                 Material **materials_ptr,
                                 MaterialDescriptor *materials_desc,
                                 SphereDescriptor *spheres_desc,
                                 int hittables_count) {

  // TODO:: one thread per object. (but only thread 0 assemles the world)
  for (int i = 0; i < hittables_count; ++i) {
    Color albedo(materials_desc[i].r, materials_desc[i].g, materials_desc[i].b);
    switch (materials_desc[i].type) {
    case MaterialType::LAMBERTIAN:
      new (materials_ptr[i]) Lambertian(albedo);
      break;
    case MaterialType::METAL:
      new (materials_ptr[i]) Metal(albedo, materials_desc[i].fuzz);
      break;
    case MaterialType::DIELECTRIC:
      new (materials_ptr[i]) Dielectric(materials_desc[i].refraction_index);
      break;
    default:
      break;
    }
  }

  for (int i = 0; i < hittables_count; ++i) {
    Point3 hittable_center(
        spheres_desc[i].x, spheres_desc[i].y, spheres_desc[i].z);
    new (spheres_ptr[i])
        Sphere(hittable_center, spheres_desc[i].radius, materials_ptr[i]);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // use placement new to initialize the world object
    new (world_ptr) HittableList(hittables_count);
    for (int i = 0; i < hittables_count; ++i) {
      world_ptr->add(spheres_ptr[i]);
    }
  }
}

__global__ void device_object_destructor(Camera *camera_ptr,
                                         HittableList *world,
                                         Sphere **spheres_ptr,
                                         Material **materials_ptr,
                                         int hittables_count) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    camera_ptr->~Camera();
    for (int i = 0; i < hittables_count; ++i) {
      spheres_ptr[i]->~Sphere();
    }

    for (int i = 0; i < hittables_count; ++i) {
      materials_ptr[i]->~Material();
    }
  }
}

void cleanup_device(Camera *camera_ptr,
                    HittableList *world_ptr,
                    Sphere **spheres_ptr,
                    Material **materials_ptr,
                    int hittables_count,
                    int *framebuffer) {

  device_object_destructor<<<1, 1>>>(
      camera_ptr, world_ptr, spheres_ptr, materials_ptr, hittables_count);
  CHECK_CUDA(cudaFree(camera_ptr));
  CHECK_CUDA(cudaFree(world_ptr));
  CHECK_CUDA(cudaFree(spheres_ptr));
  CHECK_CUDA(cudaFree(materials_ptr));
  CHECK_CUDA(cudaFree(framebuffer));
}

void cleanup_host(int *framebuffer_host,
                  Sphere **aux_spheres_ptr,
                  Material **aux_materials_ptr,
                  int hittables_count) {
  delete[] framebuffer_host;
  // All material objects have been free in `cleanup_device`.
  // So we don't need to free them here.
  delete[] aux_spheres_ptr;
  delete[] aux_materials_ptr;
}

__global__ void camera_render_launcher(Camera *camera_ptr,
                                       int *framebuffer,
                                       HittableList *world,
                                       const int image_width,
                                       const int image_height) {
  camera_ptr->render_to_framebuffer(
      framebuffer, world, image_width, image_height);
}

int main(int argc, char **argv) {
  // Parse command line arguments, read image width and height.
  int image_width = std::stoi(argv[1]);
  int image_height = std::stoi(argv[2]);
  const auto aspect_ratio = double(image_width) / image_height;
  if (std::fabs(aspect_ratio - 16.0 / 9.0) > 0.001) {
    std::cerr << "Aspect ratio other than 16:9 is not supported now"
              << std::endl;
    return -1;
  }

  // Do some initialization
  CHECK_CUDA(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024));

  const int grid_x = image_width / 16;
  const int grid_y = image_height / 9;
  const int block_x = std::min(grid_x / 16, 32);
  const int block_y = std::min(grid_y / 16, 32);
  const double vfov = 20.0;
  const int samples_per_pixel = 100;

  Camera *camera_ptr;
  // TODO: Is there any neovim plugin to show the expanded macro in a floating
  // window?

  // Allocate memory for camera
  CHECK_CUDA(cudaMalloc(&camera_ptr, sizeof(Camera)));
  CHECK_CUDA(cudaDeviceSynchronize());
  // use placement new to initialize the camera object
  initialize_camera<<<1, 1>>>(camera_ptr,
                              13.0,
                              2.0,
                              3.0,
                              0.0,
                              0.0,
                              0.0,
                              0.0,
                              1.0,
                              0.0,
                              image_width,
                              image_height,
                              samples_per_pixel,
                              vfov,
                              0.6,
                              10.0);

  // Allocate memory for the world
  HittableList *world_ptr;
  CHECK_CUDA(cudaMalloc(&world_ptr, sizeof(HittableList)));

  constexpr int hittables_count =
      1 + 3 + 22 * 22; // ground + 3 large spheres + 22 * 22 small spheres

  // Now we need to allocate memory for the objects in the world
  // We can not just use the for loop and cudaMalloc to allocate
  // memory for each object because points in `spheres_ptr` are
  // on the devcie, and we can not use them and call `cudaMalloc`
  // in the host code.
  // Therefore, we need to first allocate memory for the pointers
  // on the host (using `cudaMalloc`), that copy the pointer value
  // to the device ones.
  Sphere **aux_spheres_ptr = new Sphere *[hittables_count];
  for (int i = 0; i < hittables_count; ++i) {
    CHECK_CUDA(cudaMalloc(&aux_spheres_ptr[i], sizeof(Sphere)));
  }

  // Copy the pointers to the device
  Sphere **spheres_ptr;
  CHECK_CUDA(cudaMalloc(&spheres_ptr, hittables_count * sizeof(Sphere *)));
  CHECK_CUDA(cudaMemcpy(spheres_ptr,
                        aux_spheres_ptr,
                        hittables_count * sizeof(Sphere *),
                        cudaMemcpyHostToDevice));

  // Descriptors are needed because size of each material is different.
  MaterialDescriptor host_materials_desc[hittables_count];
  SphereDescriptor host_spheres_desc[hittables_count];
  unsigned int hittable_idx = 0;
  // ground
  host_materials_desc[hittable_idx] = {
      MaterialType::LAMBERTIAN, 0.5, 0.5, 0.5, 0.0, 0.0};
  host_spheres_desc[hittable_idx++] = {0, -1000, 0, 1000};

  for (int a = -11; a < 11; ++a) {
    for (int b = -11; b < 11; ++b) {
      auto chosen_mat = random_double_host();
      SphereDescriptor sphere_desc{a + 0.9 * random_double_host(),
                                   0.2,
                                   b + 0.9 * random_double_host(),
                                   0.2};
      host_spheres_desc[hittable_idx] = sphere_desc;
      // if (std::sqrt((sphere_desc.x - 4) * (sphere_desc.x - 4) +
      //               (sphere_desc.y - 0.2) * (sphere_desc.y - 0.2) +
      //               sphere_desc.z * sphere_desc.z) <= 0.9) {
      //   continue;
      // }

      if (chosen_mat < 0.8) {
        host_materials_desc[hittable_idx] = {
            MaterialType::LAMBERTIAN,
            random_double_host() * random_double_host(),
            random_double_host() * random_double_host(),
            random_double_host() * random_double_host(),
            0.0,
            0.0};
      } else if (chosen_mat < 0.95) {
        host_materials_desc[hittable_idx] = {
            MaterialType::METAL,
            random_double_host(0.5, 1.0),
            random_double_host(0.5, 1.0),
            random_double_host(0.5, 1.0),
            random_double_host(0, 0.5),
        };
      } else {
        host_materials_desc[hittable_idx] = {
            MaterialType::DIELECTRIC,
            0.0,
            0.0,
            0.0,
            0.0,
            1.5,
        };
      }
      hittable_idx++;
    }
  }

  // large sphere 1
  host_spheres_desc[hittable_idx] = {0.0, 1.0, 0.0, 1.0};
  host_materials_desc[hittable_idx++] = {
      MaterialType::DIELECTRIC, 0.0, 0.0, 0.0, 0.0, 1.5};
  // large sphere 2
  host_spheres_desc[hittable_idx] = {-4.0, 1.0, 0.0, 1.0};
  host_materials_desc[hittable_idx++] = {
      MaterialType::LAMBERTIAN, 0.4, 0.2, 0.1, 0.0, 0.0};
  // large sphere 3
  host_spheres_desc[hittable_idx] = {4.0, 1.0, 0.0, 1.0};
  host_materials_desc[hittable_idx++] = {
      MaterialType::METAL, 0.7, 0.6, 0.5, 0.0, 0.0};
  assert(hittable_idx == hittables_count);

  // Allocate memory for the materials according to their own type.
  // Use the same allocation logic as the spheres.
  Material **aux_materials_ptr = new Material *[hittables_count];
  for (int i = 0; i < hittables_count; ++i) {
    switch (host_materials_desc[i].type) {
    case MaterialType::LAMBERTIAN:
      CHECK_CUDA(cudaMalloc(&aux_materials_ptr[i], sizeof(Lambertian)));
      break;
    case MaterialType::METAL:
      CHECK_CUDA(cudaMalloc(&aux_materials_ptr[i], sizeof(Metal)));
      break;
    case MaterialType::DIELECTRIC:
      CHECK_CUDA(cudaMalloc(&aux_materials_ptr[i], sizeof(Dielectric)));
      break;
    default:
      std::cerr << "Unknown material type" << std::endl;
      return -1;
    }
  }

  // Copy the pointers to device.
  Material **materials_ptr;
  CHECK_CUDA(cudaMalloc(&materials_ptr, hittables_count * sizeof(Material *)));
  CHECK_CUDA(cudaMemcpy(materials_ptr,
                        aux_materials_ptr,
                        hittables_count * sizeof(Material *),
                        cudaMemcpyHostToDevice));

  // Copy the material descriptors to the device.
  // So we can initialize the materials on device.
  MaterialDescriptor *device_materials_desc;
  CHECK_CUDA(cudaMalloc(&device_materials_desc, sizeof(host_materials_desc)));
  CHECK_CUDA(cudaMemcpy(device_materials_desc,
                        host_materials_desc,
                        sizeof(host_materials_desc),
                        cudaMemcpyHostToDevice));

  SphereDescriptor *device_spheres_desc;
  CHECK_CUDA(cudaMalloc(&device_spheres_desc, sizeof(host_spheres_desc)));
  CHECK_CUDA(cudaMemcpy(device_spheres_desc,
                        host_spheres_desc,
                        sizeof(host_spheres_desc),
                        cudaMemcpyHostToDevice));

  // Initialize the world object by placement new.
  // Add objects to the world.
  initialize_world<<<1, 1>>>(world_ptr,
                             spheres_ptr,
                             materials_ptr,
                             device_materials_desc,
                             device_spheres_desc,
                             hittables_count);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Allocate memory for the framebuffer
  int *framebuffer;
  CHECK_CUDA(
      cudaMalloc(&framebuffer, image_width * image_height * 3 * sizeof(int)));

  // Initialize the `curand` states.
  init_random(1234ULL, grid_x, grid_y, block_x, block_y);
  // End all initialization

  dim3 grid(grid_x, grid_y);
  dim3 block(block_x, block_y);
  std::clog << "Begin rendering" << std::endl;

  camera_render_launcher<<<grid, block>>>(
      camera_ptr, framebuffer, world_ptr, image_width, image_height);

  CHECK_CUDA(cudaDeviceSynchronize());
  std::clog << "End rendering" << std::endl;

  std::clog << "Begin writing to file" << std::endl;
  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

  int *framebuffer_host = new int[image_width * image_height * 3];
  CHECK_CUDA(cudaMemcpy(framebuffer_host,
                        framebuffer,
                        image_width * image_height * 3 * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());
  for (int row = 0; row < image_height; row++) {
    for (int col = 0; col < image_width; col++) {
      auto pixel_index = (row * image_width + col) * 3;
      std::cout << framebuffer_host[pixel_index + 0] << " "
                << framebuffer_host[pixel_index + 1] << " "
                << framebuffer_host[pixel_index + 2] << "\n";
    }
  }
  std::clog << "End writing to file" << std::endl;

  // TODO: free memory

  cleanup_device(camera_ptr,
                 world_ptr,
                 spheres_ptr,
                 materials_ptr,
                 hittables_count,
                 framebuffer);
  cleanup_host(
      framebuffer_host, aux_spheres_ptr, aux_materials_ptr, hittables_count);
  cleanup_random();

  return 0;
}
