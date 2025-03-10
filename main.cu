#include "camera.cuh"
#include "hittable_list.cuh"
#include "material.cuh"
#include "random.cuh"
#include "sphere.cuh"
#include "utils.cuh"
#include <cmath>
#include <cstdio>
#include <iostream>

__global__ void initialize_camera(Camera *camera_ptr, double camera_x,
                                  double camera_y, double camera_z,
                                  const double focal_length, int image_width,
                                  int image_height, double samples_per_pixel,
                                  double vfov) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (camera_ptr) Camera();
    camera_ptr->intialize(Vec3(camera_x, camera_y, camera_z), focal_length,
                          image_width, image_height, samples_per_pixel, vfov);
  }
}

__global__ void initialize_world(HittableList *world_ptr, Sphere **spheres_ptr,
                                 Material **materials_ptr,
                                 MaterialDescriptor *materials_desc,
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

  new (spheres_ptr[0]) Sphere(Point3(0, -100.5, -1.0), 100, materials_ptr[0]);
  new (spheres_ptr[1]) Sphere(Point3(0, 0, -1.2), 0.5, materials_ptr[1]);
  new (spheres_ptr[2]) Sphere(Point3(-1.0, 0, -1.0), 0.5, materials_ptr[2]);
  new (spheres_ptr[3]) Sphere(Point3(-1.0, 0, -1.0), 0.4, materials_ptr[3]);
  new (spheres_ptr[4]) Sphere(Point3(1.0, 0, -1.0), 0.5, materials_ptr[4]);

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

void cleanup_device(Camera *camera_ptr, HittableList *world_ptr,
                    Sphere **spheres_ptr, Material **materials_ptr,
                    int hittables_count, int *framebuffer) {

  device_object_destructor<<<1, 1>>>(camera_ptr, world_ptr, spheres_ptr,
                                     materials_ptr, hittables_count);
  CHECK_CUDA(cudaFree(camera_ptr));
  CHECK_CUDA(cudaFree(world_ptr));
  CHECK_CUDA(cudaFree(spheres_ptr));
  CHECK_CUDA(cudaFree(materials_ptr));
  CHECK_CUDA(cudaFree(framebuffer));
}

void cleanup_host(int *framebuffer_host, Sphere **aux_spheres_ptr,
                  Material **aux_materials_ptr, int hittables_count) {
  delete[] framebuffer_host;
  // All material objects have been free in `cleanup_device`.
  // So we don't need to free them here.
  delete[] aux_spheres_ptr;
  delete[] aux_materials_ptr;
}

__global__ void camera_render_launcher(Camera *camera_ptr, int *framebuffer,
                                       HittableList *world,
                                       const int image_width,
                                       const int image_height) {
  camera_ptr->render_to_framebuffer(framebuffer, world, image_width,
                                    image_height);
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
  const double focal_length = 1.0;
  const double vfov = 90.0;

  Camera *camera_ptr;
  // TODO: Is there any neovim plugin to show the expanded macro in a floating
  // window?

  // Allocate memory for camera
  CHECK_CUDA(cudaMalloc(&camera_ptr, sizeof(Camera)));
  CHECK_CUDA(cudaDeviceSynchronize());
  // use placement new to initialize the camera object
  initialize_camera<<<1, 1>>>(camera_ptr, 0.0, 0.0, 0.0, focal_length,
                              image_width, image_height, 100, vfov);

  // Allocate memory for the world
  HittableList *world_ptr;
  CHECK_CUDA(cudaMalloc(&world_ptr, sizeof(HittableList)));

  constexpr int hittables_count = 5;

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
  CHECK_CUDA(cudaMemcpy(spheres_ptr, aux_spheres_ptr,
                        hittables_count * sizeof(Sphere *),
                        cudaMemcpyHostToDevice));

  // Descriptors are needed because size of each material is different.
  const MaterialDescriptor host_materials_desc[hittables_count] = {
      {MaterialType::LAMBERTIAN, 0.8, 0.8, 0.0, 0.0, 0.0},
      {MaterialType::LAMBERTIAN, 0.1, 0.2, 0.5, 0.0, 0.0},
      {MaterialType::DIELECTRIC, 0.0, 0.0, 0.0, 0.0, 1.50},
      {MaterialType::DIELECTRIC, 0.0, 0.0, 0.0, 0.0, 1.0 / 1.5},
      {MaterialType::METAL, 0.8, 0.6, 0.2, 1.0, 0.0},
  };

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
  CHECK_CUDA(cudaMemcpy(materials_ptr, aux_materials_ptr,
                        hittables_count * sizeof(Material *),
                        cudaMemcpyHostToDevice));

  // Copy the material descriptors to the device.
  // So we can initialize the materials on device.
  MaterialDescriptor *device_materials_desc;
  CHECK_CUDA(cudaMalloc(&device_materials_desc, sizeof(host_materials_desc)));
  CHECK_CUDA(cudaMemcpy(device_materials_desc, host_materials_desc,
                        sizeof(host_materials_desc), cudaMemcpyHostToDevice));

  // Initialize the world object by placement new.
  // Add objects to the world.
  initialize_world<<<1, 1>>>(world_ptr, spheres_ptr, materials_ptr,
                             device_materials_desc, hittables_count);
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

  camera_render_launcher<<<grid, block>>>(camera_ptr, framebuffer, world_ptr,
                                          image_width, image_height);

  CHECK_CUDA(cudaDeviceSynchronize());
  std::clog << "End rendering" << std::endl;

  std::clog << "Begin writing to file" << std::endl;
  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

  int *framebuffer_host = new int[image_width * image_height * 3];
  CHECK_CUDA(cudaMemcpy(framebuffer_host, framebuffer,
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

  cleanup_device(camera_ptr, world_ptr, spheres_ptr, materials_ptr,
                 hittables_count, framebuffer);
  cleanup_host(framebuffer_host, aux_spheres_ptr, aux_materials_ptr,
               hittables_count);
  cleanup_random();

  return 0;
}
