#include "raylib.h"
#include "gjk_integration.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

// Include GPU headers at the TOP, not inside functions
#ifdef USE_CUDA
#include "gpu_gjk_interface.h"
#endif

// Define Physics Object structure
typedef struct {
    Vector3f position;
    Vector3f velocity;
    Vector3f acceleration;
    float radius;
    float mass;
    Color color;
    char name[32];
} PhysicsObject;

// GPU GJK context and mode (will be conditionally compiled)
typedef enum { MODE_CPU, MODE_GPU } GJKMode;

// Convert our Vector3f to Raylib's Vector3
Vector3 toRaylibVector3(Vector3f v) {
    return (Vector3){ v.x, v.y, v.z };
}

// Basic camera controls (simplified - no mouse rotation)
void UpdateCameraCustom(Camera3D* camera) {
    // Keyboard movement
    if (IsKeyDown(KEY_W)) camera->position.z -= 0.5f;
    if (IsKeyDown(KEY_S)) camera->position.z += 0.5f;
    if (IsKeyDown(KEY_A)) camera->position.x -= 0.5f;
    if (IsKeyDown(KEY_D)) camera->position.x += 0.5f;
    if (IsKeyDown(KEY_Q)) camera->position.y -= 0.5f;
    if (IsKeyDown(KEY_E)) camera->position.y += 0.5f;
    
    // Reset camera
    if (IsKeyPressed(KEY_R)) {
        camera->position = (Vector3){ 0.0f, 10.0f, 20.0f };
        camera->target = (Vector3){ 0.0f, 0.0f, 0.0f };
        camera->up = (Vector3){ 0.0f, 1.0f, 0.0f };
    }
}

// Simple elastic collision response
void resolve_collision(PhysicsObject* obj1, PhysicsObject* obj2) {
    // Calculate collision normal
    Vector3f delta = {
        obj2->position.x - obj1->position.x,
        obj2->position.y - obj1->position.y,
        obj2->position.z - obj1->position.z
    };
    
    float distance = sqrt(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
    if (distance == 0) return;
    
    Vector3f normal = { delta.x/distance, delta.y/distance, delta.z/distance };
    
    // Separate objects to prevent sticking
    float overlap = (obj1->radius + obj2->radius) - distance;
    if (overlap > 0) {
        obj1->position.x -= normal.x * overlap * 0.5f;
        obj1->position.y -= normal.y * overlap * 0.5f;
        obj1->position.z -= normal.z * overlap * 0.5f;
        
        obj2->position.x += normal.x * overlap * 0.5f;
        obj2->position.y += normal.y * overlap * 0.5f;
        obj2->position.z += normal.z * overlap * 0.5f;
    }
    
    // Relative velocity
    Vector3f relative_vel = {
        obj2->velocity.x - obj1->velocity.x,
        obj2->velocity.y - obj1->velocity.y,
        obj2->velocity.z - obj1->velocity.z
    };
    
    // Velocity along normal
    float vel_along_normal = relative_vel.x * normal.x + relative_vel.y * normal.y + relative_vel.z * normal.z;
    
    // Don't resolve if objects are moving apart
    if (vel_along_normal > 0) return;
    
    // Collision impulse (simplified elastic collision)
    float restitution = 0.8f; // Bounciness
    float j = -(1 + restitution) * vel_along_normal;
    j /= (1/obj1->mass + 1/obj2->mass);
    
    // Apply impulse
    Vector3f impulse = { j * normal.x, j * normal.y, j * normal.z };
    
    obj1->velocity.x -= impulse.x / obj1->mass;
    obj1->velocity.y -= impulse.y / obj1->mass;
    obj1->velocity.z -= impulse.z / obj1->mass;
    
    obj2->velocity.x += impulse.x / obj2->mass;
    obj2->velocity.y += impulse.y / obj2->mass;
    obj2->velocity.z += impulse.z / obj2->mass;
}

int main(void) {
    // Initialize window
    const int screenWidth = 1200;
    const int screenHeight = 800;
    InitWindow(screenWidth, screenHeight, "Physics Simulation - GPU GJK Collision Detection");
    
    // Initialize camera
    Camera3D camera = { 0 };
    camera.position = (Vector3){ 0.0f, 10.0f, 20.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    
    // Create physics objects (spheres)
    PhysicsObject sphere1 = {
        .position = { -5.0f, 8.0f, 0.0f },
        .velocity = { 2.0f, 0.0f, 0.0f },
        .acceleration = { 0.0f, -9.8f, 0.0f }, // Gravity
        .radius = 1.5f,
        .mass = 1.0f,
        .color = RED,
        .name = "Sphere A"
    };
    
    PhysicsObject sphere2 = {
        .position = { 5.0f, 8.0f, 0.0f },
        .velocity = { -2.0f, 0.0f, 0.0f },
        .acceleration = { 0.0f, -9.8f, 0.0f }, // Gravity
        .radius = 1.5f,
        .mass = 1.0f,
        .color = BLUE,
        .name = "Sphere B"
    };
    
    // Create GJK shapes (bounding spheres)
    GJK_Shape gjk_sphere1 = create_cube_shape(sphere1.position, sphere1.radius * 2.0f); // Approximate sphere with cube
    GJK_Shape gjk_sphere2 = create_cube_shape(sphere2.position, sphere2.radius * 2.0f);
    
    // Physics simulation variables
    bool collision = false;
    float distance = 0.0f;
    float deltaTime = 1.0f / 60.0f; // Fixed timestep for stable physics
    
    // GPU support variables
    bool gpu_available = false;
    GJKMode current_mode = MODE_GPU; // Start with GPU mode
    double cpu_time = 0, gpu_time = 0;
    
    // Try to initialize GPU if available
#ifdef USE_CUDA
    GPU_GJK_Context* gpu_context = NULL;
    gpu_available = gpu_gjk_init(&gpu_context, 10);  // Support up to 10 objects
    
    if (gpu_available) {
        // Register shapes once at startup
        gpu_gjk_register_shape(gpu_context, &gjk_sphere1, 0);
        gpu_gjk_register_shape(gpu_context, &gjk_sphere2, 1);
        current_mode = MODE_GPU;
        printf("GPU Physics Simulation Initialized!\n");
    } else {
        printf("GPU not available, using CPU physics\n");
    }
#endif
    
    SetTargetFPS(60);
    
    // Main game loop
    while (!WindowShouldClose()) {
        // Physics Update
        // Update velocities with acceleration (gravity)
        sphere1.velocity.x += sphere1.acceleration.x * deltaTime;
        sphere1.velocity.y += sphere1.acceleration.y * deltaTime;
        sphere1.velocity.z += sphere1.acceleration.z * deltaTime;
        
        sphere2.velocity.x += sphere2.acceleration.x * deltaTime;
        sphere2.velocity.y += sphere2.acceleration.y * deltaTime;
        sphere2.velocity.z += sphere2.acceleration.z * deltaTime;
        
        // Update positions with velocity
        sphere1.position.x += sphere1.velocity.x * deltaTime;
        sphere1.position.y += sphere1.velocity.y * deltaTime;
        sphere1.position.z += sphere1.velocity.z * deltaTime;
        
        sphere2.position.x += sphere2.velocity.x * deltaTime;
        sphere2.position.y += sphere2.velocity.y * deltaTime;
        sphere2.position.z += sphere2.velocity.z * deltaTime;
        
        // Ground collision
        if (sphere1.position.y - sphere1.radius < 0) {
            sphere1.position.y = sphere1.radius;
            sphere1.velocity.y = -sphere1.velocity.y * 0.8f; // Bounce with damping
        }
        if (sphere2.position.y - sphere2.radius < 0) {
            sphere2.position.y = sphere2.radius;
            sphere2.velocity.y = -sphere2.velocity.y * 0.8f; // Bounce with damping
        }
        
        // Wall collisions (simple boundary)
        float boundary = 15.0f;
        if (fabs(sphere1.position.x) > boundary - sphere1.radius) {
            sphere1.velocity.x = -sphere1.velocity.x * 0.8f;
            sphere1.position.x = (sphere1.position.x > 0) ? boundary - sphere1.radius : -boundary + sphere1.radius;
        }
        if (fabs(sphere2.position.x) > boundary - sphere2.radius) {
            sphere2.velocity.x = -sphere2.velocity.x * 0.8f;
            sphere2.position.x = (sphere2.position.x > 0) ? boundary - sphere2.radius : -boundary + sphere2.radius;
        }
        
        // Update GJK shape positions
        gjk_sphere1.position = sphere1.position;
        gjk_sphere2.position = sphere2.position;
        
        // Mode switching (only if GPU available)
        if (IsKeyPressed(KEY_TAB) && gpu_available) {
            current_mode = (current_mode == MODE_CPU) ? MODE_GPU : MODE_CPU;
            printf("Switched to %s physics\n", current_mode == MODE_GPU ? "GPU" : "CPU");
        }
        
        // Reset simulation
        if (IsKeyPressed(KEY_SPACE)) {
            sphere1.position = (Vector3f){ -5.0f, 8.0f, 0.0f };
            sphere1.velocity = (Vector3f){ 2.0f, 0.0f, 0.0f };
            sphere2.position = (Vector3f){ 5.0f, 8.0f, 0.0f };
            sphere2.velocity = (Vector3f){ -2.0f, 0.0f, 0.0f };
            printf("Simulation Reset!\n");
        }
        
        // Collision detection with timing
        clock_t start, end;
        
#ifdef USE_CUDA
        if (current_mode == MODE_GPU && gpu_available) {
            start = clock();
            
            // Update positions on GPU
            gpu_gjk_update_position(gpu_context, 0, sphere1.position);
            gpu_gjk_update_position(gpu_context, 1, sphere2.position);
            
            // Check collision on GPU
            int pairs[] = {0, 1};
            bool gpu_results[1];
            gpu_gjk_batch_check(gpu_context, pairs, 1, gpu_results);
            collision = gpu_results[0];
            
            end = clock();
            gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
        } else {
#endif
            // CPU collision detection
            start = clock();
            collision = openGJK_collision_cpu(&gjk_sphere1, &gjk_sphere2);
            end = clock();
            cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
#ifdef USE_CUDA
        }
#endif
        
        // Handle collision response
        if (collision) {
            resolve_collision(&sphere1, &sphere2);
        }
        
        // Calculate distance for display
        float dx = sphere1.position.x - sphere2.position.x;
        float dy = sphere1.position.y - sphere2.position.y;
        float dz = sphere1.position.z - sphere2.position.z;
        distance = sqrt(dx*dx + dy*dy + dz*dz);
        
        // Update camera
        UpdateCameraCustom(&camera);
        
        // Drawing
        BeginDrawing();
            ClearBackground(RAYWHITE);
            
            BeginMode3D(camera);
            
            // Draw floor
            DrawPlane((Vector3){0, 0, 0}, (Vector2){50, 50}, LIGHTGRAY);
            
            // Draw spheres with collision highlighting
            Color sphere1Color = collision ? YELLOW : RED;
            Color sphere2Color = collision ? YELLOW : BLUE;
            
            Vector3 pos1 = toRaylibVector3(sphere1.position);
            Vector3 pos2 = toRaylibVector3(sphere2.position);
            
            DrawSphere(pos1, sphere1.radius, sphere1Color);
            DrawSphere(pos2, sphere2.radius, sphere2Color);
            
            // Draw sphere outlines
            DrawSphereWires(pos1, sphere1.radius, 8, 8, DARKGRAY);
            DrawSphereWires(pos2, sphere2.radius, 8, 8, DARKGRAY);
            
            // Draw coordinate axes
            DrawLine3D((Vector3){0,0,0}, (Vector3){5,0,0}, RED);   // X axis
            DrawLine3D((Vector3){0,0,0}, (Vector3){0,5,0}, GREEN); // Y axis
            DrawLine3D((Vector3){0,0,0}, (Vector3){0,0,5}, BLUE);  // Z axis
            
            EndMode3D();
            
            // Draw UI
            DrawText("Physics Simulation - GPU GJK Collision Detection", 10, 10, 20, DARKGRAY);
            
            // Mode and timing information
#ifdef USE_CUDA
            if (gpu_available) {
                const char* mode_text = (current_mode == MODE_GPU) ? "GPU PHYSICS" : "CPU PHYSICS";
                Color mode_color = (current_mode == MODE_GPU) ? GREEN : BLUE;
                DrawText(mode_text, screenWidth - 150, 10, 20, mode_color);
                DrawText("Press TAB to switch modes", 10, 40, 20, DARKGRAY);
                
                char timing_text[100];
                if (current_mode == MODE_GPU) {
                    sprintf(timing_text, "GPU Physics Time: %.3f ms", gpu_time);
                    DrawText(timing_text, 10, 70, 20, GREEN);
                } else {
                    sprintf(timing_text, "CPU Physics Time: %.3f ms", cpu_time);
                    DrawText(timing_text, 10, 70, 20, BLUE);
                }
            } else {
                DrawText("CPU PHYSICS (GPU not available)", screenWidth - 280, 10, 20, BLUE);
                char timing_text[100];
                sprintf(timing_text, "CPU Time: %.3f ms", cpu_time);
                DrawText(timing_text, 10, 40, 20, BLUE);
            }
#else
            DrawText("CPU PHYSICS", screenWidth - 150, 10, 20, BLUE);
            char timing_text[100];
            sprintf(timing_text, "CPU Time: %.3f ms", cpu_time);
            DrawText(timing_text, 10, 40, 20, BLUE);
#endif
            
            // Physics info
            if (collision) {
                DrawText("COLLISION! - BOUNCING", 10, 100, 30, RED);
            } else {
                DrawText("No collision", 10, 100, 30, GREEN);
            }
            
            char distanceText[50], velocity1Text[80], velocity2Text[80];
            sprintf(distanceText, "Distance: %.2f", distance);
            sprintf(velocity1Text, "Red Vel: (%.1f, %.1f, %.1f)", sphere1.velocity.x, sphere1.velocity.y, sphere1.velocity.z);
            sprintf(velocity2Text, "Blue Vel: (%.1f, %.1f, %.1f)", sphere2.velocity.x, sphere2.velocity.y, sphere2.velocity.z);
            
            DrawText(distanceText, 10, 140, 20, DARKGRAY);
            DrawText(velocity1Text, 10, 170, 18, RED);
            DrawText(velocity2Text, 10, 195, 18, BLUE);
            
            // Controls
            DrawText("Controls:", 10, 230, 20, DARKGRAY);
            DrawText("SPACE: Reset simulation", 10, 260, 18, DARKGRAY);
            DrawText("TAB: Switch CPU/GPU mode", 10, 285, 18, DARKGRAY);
            DrawText("WASD+QE: Move camera | R: Reset camera", 10, 310, 18, DARKGRAY);
            
            // Physics features
            DrawText("Physics Features:", 10, 345, 18, DARKGRAY);
            DrawText("- Gravity simulation", 10, 370, 16, DARKGRAY);
            DrawText("- Elastic collisions", 10, 390, 16, DARKGRAY);
            DrawText("- Ground and wall bounds", 10, 410, 16, DARKGRAY);
            DrawText("- Real-time collision response", 10, 430, 16, DARKGRAY);
            
            // Draw coordinates
            char pos1Text[80], pos2Text[80];
            sprintf(pos1Text, "Red: (%.1f, %.1f, %.1f)", sphere1.position.x, sphere1.position.y, sphere1.position.z);
            sprintf(pos2Text, "Blue: (%.1f, %.1f, %.1f)", sphere2.position.x, sphere2.position.y, sphere2.position.z);
            
            DrawText(pos1Text, 10, screenHeight - 60, 18, RED);
            DrawText(pos2Text, 10, screenHeight - 35, 18, BLUE);
            
        EndDrawing();
    }
    
    // Cleanup
#ifdef USE_CUDA
    if (gpu_available) {
        gpu_gjk_cleanup(&gpu_context);
    }
#endif
    free_shape(&gjk_sphere1);
    free_shape(&gjk_sphere2);
    CloseWindow();
    
    return 0;
}