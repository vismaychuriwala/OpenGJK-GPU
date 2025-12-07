#ifndef CAMERA_H
#define CAMERA_H

#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

struct Camera3D {
    glm::vec3 position;
    glm::vec3 target;
    glm::vec3 up;
    float fovy;  // Field of view Y in degrees

    // Cached matrices
    glm::mat4 view_matrix;
    glm::mat4 projection_matrix;
};

// Initialize camera with default values
void camera_init(Camera3D* camera, glm::vec3 position, glm::vec3 target, float fovy);

// Update camera matrices (call after modifying position/target)
void camera_update_matrices(Camera3D* camera, float aspect_ratio);

// Update camera based on user input (ported from main.c)
void camera_update_controls(Camera3D* camera);

// Reset camera to default position
void camera_reset(Camera3D* camera);

#endif // CAMERA_H
