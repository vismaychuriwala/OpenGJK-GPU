#include "camera.h"
#include "input.h"
#include <cmath>
#include <GLFW/glfw3.h>

void camera_init(Camera3D* camera, glm::vec3 position, glm::vec3 target, float fovy) {
    camera->position = position;
    camera->target = target;
    camera->up = glm::vec3(0.0f, 1.0f, 0.0f);
    camera->fovy = fovy;
    camera->view_matrix = glm::mat4(1.0f);
    camera->projection_matrix = glm::mat4(1.0f);
}

void camera_update_matrices(Camera3D* camera, float aspect_ratio) {
    // View matrix
    camera->view_matrix = glm::lookAt(camera->position, camera->target, camera->up);

    // Projection matrix
    camera->projection_matrix = glm::perspective(
        glm::radians(camera->fovy), aspect_ratio, 0.1f, 1000.0f);
}

void camera_update_controls(Camera3D* camera) {
    // Calculate current position relative to target
    float dx = camera->position.x - camera->target.x;
    float dy = camera->position.y - camera->target.y;
    float dz = camera->position.z - camera->target.z;

    // Mouse rotation (LEFT mouse button) - orbit around target
    if (IsMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT)) {
        float mouseDeltaX, mouseDeltaY;
        GetMouseDelta(&mouseDeltaX, &mouseDeltaY);

        // Horizontal rotation (Y-axis)
        float angleH = mouseDeltaX * 0.003f;
        float cosH = std::cos(-angleH);
        float sinH = std::sin(-angleH);
        float newX = dx * cosH - dz * sinH;
        float newZ = dx * sinH + dz * cosH;
        dx = newX;
        dz = newZ;

        // Vertical rotation (simple pitch with clamping)
        float angleV = mouseDeltaY * 0.003f;
        float radius = std::sqrt(dx*dx + dy*dy + dz*dz);
        float currentPitch = std::asin(dy / radius);
        float newPitch = currentPitch + angleV;

        // Clamp pitch to avoid gimbal lock
        if (newPitch > -1.5f && newPitch < 1.5f) {
            float horizontalDist = std::sqrt(dx*dx + dz*dz);
            dy = radius * std::sin(newPitch);
            float newHorizontalDist = radius * std::cos(newPitch);
            float scale = newHorizontalDist / horizontalDist;
            dx *= scale;
            dz *= scale;
        }
    }

    // Arrow key rotation
    float rotSpeed = 0.02f;

    // Left/Right arrows - horizontal rotation
    if (IsKeyDown(GLFW_KEY_LEFT)) {
        float angleH = rotSpeed;
        float cosH = std::cos(-angleH);
        float sinH = std::sin(-angleH);
        float newX = dx * cosH - dz * sinH;
        float newZ = dx * sinH + dz * cosH;
        dx = newX;
        dz = newZ;
    }
    if (IsKeyDown(GLFW_KEY_RIGHT)) {
        float angleH = -rotSpeed;
        float cosH = std::cos(-angleH);
        float sinH = std::sin(-angleH);
        float newX = dx * cosH - dz * sinH;
        float newZ = dx * sinH + dz * cosH;
        dx = newX;
        dz = newZ;
    }

    // Up/Down arrows - vertical rotation
    if (IsKeyDown(GLFW_KEY_UP) || IsKeyDown(GLFW_KEY_DOWN)) {
        float angleV = IsKeyDown(GLFW_KEY_UP) ? -rotSpeed : rotSpeed;
        float radius = std::sqrt(dx*dx + dy*dy + dz*dz);
        float currentPitch = std::asin(dy / radius);
        float newPitch = currentPitch + angleV;

        // Clamp pitch to avoid gimbal lock
        if (newPitch > -1.5f && newPitch < 1.5f) {
            float horizontalDist = std::sqrt(dx*dx + dz*dz);
            dy = radius * std::sin(newPitch);
            float newHorizontalDist = radius * std::cos(newPitch);
            float scale = newHorizontalDist / horizontalDist;
            dx *= scale;
            dz *= scale;
        }
    }

    // Update camera position
    camera->position.x = camera->target.x + dx;
    camera->position.y = camera->target.y + dy;
    camera->position.z = camera->target.z + dz;

    // Keyboard movement (WASD controls in world coordinates)
    // Move both camera and target to pan the view without changing rotation
    float moveSpeed = 0.5f;
    glm::vec3 movement(0.0f);

    if (IsKeyDown(GLFW_KEY_W)) movement.z -= moveSpeed;  // Forward in world -Z
    if (IsKeyDown(GLFW_KEY_S)) movement.z += moveSpeed;  // Backward in world +Z
    if (IsKeyDown(GLFW_KEY_A)) movement.x -= moveSpeed;  // Left in world -X
    if (IsKeyDown(GLFW_KEY_D)) movement.x += moveSpeed;  // Right in world +X
    if (IsKeyDown(GLFW_KEY_Q)) movement.y -= moveSpeed;  // Down in world -Y
    if (IsKeyDown(GLFW_KEY_E)) movement.y += moveSpeed;  // Up in world +Y

    // Apply movement to both camera and target (maintains orbit orientation)
    camera->position += movement;
    camera->target += movement;

    // Mouse scroll for zoom (move camera toward/away from target)
    float scroll = GetMouseWheelMove();
    if (scroll != 0.0f) {
        // Calculate direction from camera to target
        glm::vec3 zoomDir = camera->target - camera->position;
        float distance = glm::length(zoomDir);

        // Normalize direction
        if (distance > 0.1f) {  // Prevent division by zero
            zoomDir = glm::normalize(zoomDir);

            // Zoom speed (scales with current distance for smooth feel)
            float zoomSpeed = distance * 0.1f;
            float zoomAmount = scroll * zoomSpeed;

            // Don't zoom too close (min distance = 2 units)
            if (distance - zoomAmount > 2.0f || zoomAmount < 0.0f) {
                camera->position += zoomDir * zoomAmount;
            }
        }
    }
}

void camera_reset(Camera3D* camera) {
    // Get boundary from sim_config.h (assuming BOUNDARY = 30.0f)
    float boundary = 30.0f;
    camera->position = glm::vec3(0.0f, boundary * 0.8f, boundary * 1.6f);
    camera->target = glm::vec3(0.0f, -2.0f, 0.0f);
    camera->up = glm::vec3(0.0f, 1.0f, 0.0f);
}
