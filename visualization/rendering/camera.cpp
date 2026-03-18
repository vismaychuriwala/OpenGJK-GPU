#include "camera.h"
#include "input.h"
#include <cmath>
#include <algorithm>
#include <GLFW/glfw3.h>

void camera_init(Camera3D* camera, glm::vec3 position, glm::vec3 target, float fovy) {
    camera->position = position;
    camera->target = target;
    camera->up = glm::vec3(0.0f, 1.0f, 0.0f);
    camera->fovy = fovy;
    camera->boundary = 30.0f;  // default, overwritten by camera_reset
    camera->view_matrix = glm::mat4(1.0f);
    camera->projection_matrix = glm::mat4(1.0f);
}

void camera_update_matrices(Camera3D* camera, float aspect_ratio) {
    camera->view_matrix = glm::lookAt(camera->position, camera->target, camera->up);

    float far_clip  = fmaxf(100.0f, fminf(camera->boundary * 4.0f, 50000.0f));
    float near_clip = fmaxf(0.01f, fminf(far_clip * 0.0001f, 1.0f));
    camera->projection_matrix = glm::perspective(
        glm::radians(camera->fovy), aspect_ratio, near_clip, far_clip);
}

void camera_update_controls(Camera3D* camera, float deltaTime) {
    // Work with the normalised forward direction; preserve distance for scroll-zoom.
    glm::vec3 fwd = camera->target - camera->position;
    float dist = glm::length(fwd);
    if (dist < 1e-6f) dist = 1e-6f;
    fwd /= dist;

    static const glm::vec3 WORLD_UP(0.0f, 1.0f, 0.0f);

    // Yaw: rotate fwd around world Y.
    auto yaw = [&](float angle) {
        float c = std::cos(angle), s = std::sin(angle);
        float nx = fwd.x * c - fwd.z * s;
        float nz = fwd.x * s + fwd.z * c;
        fwd.x = nx;
        fwd.z = nz;
        fwd = glm::normalize(fwd);
    };

    // Pitch: tilt fwd up/down, clamped away from vertical.
    auto pitch = [&](float angle) {
        float curPitch = std::asin(std::fmax(-1.0f, std::fmin(1.0f, fwd.y)));
        float newPitch = std::fmax(-1.5f, std::fmin(1.5f, curPitch + angle));
        float hx = fwd.x, hz = fwd.z;
        float hMag = std::sqrt(hx * hx + hz * hz);
        if (hMag > 1e-6f) { hx /= hMag; hz /= hMag; }
        fwd.x = hx * std::cos(newPitch);
        fwd.y = std::sin(newPitch);
        fwd.z = hz * std::cos(newPitch);
    };

    // Mouse drag: first-person look (pivots around camera position, not world origin)
    if (IsMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT)) {
        float mdx, mdy;
        GetMouseDelta(&mdx, &mdy);
        yaw  ( mdx * 0.003f);
        pitch(-mdy * 0.003f);  // negative: screen-Y down = look down
    }

    // Arrow keys
    float rotSpeed = 0.02f;
    if (IsKeyDown(GLFW_KEY_LEFT))  yaw  (-rotSpeed);
    if (IsKeyDown(GLFW_KEY_RIGHT)) yaw  ( rotSpeed);
    if (IsKeyDown(GLFW_KEY_UP))    pitch( rotSpeed);
    if (IsKeyDown(GLFW_KEY_DOWN))  pitch(-rotSpeed);

    // Commit new look direction; target floats in front of the camera.
    camera->target = camera->position + fwd * dist;

    // WASD: camera-relative translation, speed proportional to scene size
    float moveSpeed = camera->boundary * 0.5f * deltaTime;
    glm::vec3 right = glm::normalize(glm::cross(fwd, WORLD_UP));

    glm::vec3 movement(0.0f);
    if (IsKeyDown(GLFW_KEY_W)) movement += fwd      *  moveSpeed;
    if (IsKeyDown(GLFW_KEY_S)) movement += fwd      * -moveSpeed;
    if (IsKeyDown(GLFW_KEY_A)) movement += right    * -moveSpeed;
    if (IsKeyDown(GLFW_KEY_D)) movement += right    *  moveSpeed;
    if (IsKeyDown(GLFW_KEY_Q)) movement += WORLD_UP * -moveSpeed;
    if (IsKeyDown(GLFW_KEY_E)) movement += WORLD_UP *  moveSpeed;

    camera->position += movement;
    camera->target   += movement;

    // Scroll: dolly toward/away from the current target point
    float scroll = std::fmax(-3.0f, std::fmin(3.0f, GetMouseWheelMove()));
    if (scroll != 0.0f) {
        float zoomSpeed  = std::max(dist * 0.1f, 1.0f);
        float zoomAmount = scroll * zoomSpeed;
        if (dist - zoomAmount < 0.1f) zoomAmount = dist - 0.1f;
        camera->position += fwd * zoomAmount;
        // target stays fixed: scroll zooms toward whatever you're looking at
    }
}

void camera_reset(Camera3D* camera, float boundary) {
    camera->boundary = boundary;
    camera->position = glm::vec3(0.0f, boundary * 0.8f, boundary * 1.6f);
    camera->target = glm::vec3(0.0f, -2.0f, 0.0f);
    camera->up = glm::vec3(0.0f, 1.0f, 0.0f);
}
