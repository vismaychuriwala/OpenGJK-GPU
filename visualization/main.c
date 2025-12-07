#include "raylib.h"
#include "gjk_integration.h"
#include "sim_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

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

// Draw a solid icosahedron polytope with wireframe edges
void draw_polytope_solid(const GJK_Shape* shape, Color color) {
    if (!shape || shape->num_vertices != 12) return;

    // Icosahedron has 20 triangular faces
    // Define faces using vertex indices (ccw winding for outward normals)
    static const int faces[20][3] = {
        {0, 8, 4},   {0, 4, 6},   {0, 6, 9},   {0, 9, 2},   {0, 2, 8},
        {1, 4, 10},  {1, 6, 4},   {1, 11, 6},  {1, 9, 11},  {1, 10, 9},
        {2, 5, 8},   {2, 7, 5},   {2, 9, 7},   {3, 10, 5},  {3, 5, 7},
        {3, 7, 11},  {3, 11, 10}, {4, 8, 10},  {5, 10, 8},  {6, 11, 9}
    };

    // Icosahedron has 30 edges
    static const int edges[30][2] = {
        {0, 2}, {0, 4}, {0, 6}, {0, 8}, {0, 9},
        {1, 4}, {1, 6}, {1, 9}, {1, 10}, {1, 11},
        {2, 5}, {2, 7}, {2, 8}, {2, 9},
        {3, 5}, {3, 7}, {3, 10}, {3, 11},
        {4, 6}, {4, 8}, {4, 10},
        {5, 7}, {5, 8}, {5, 10},
        {6, 9}, {6, 11},
        {7, 9}, {7, 11},
        {8, 10},
        {9, 11}
    };

    // Transform vertices to world space
    Vector3 world_verts[12];
    for (int i = 0; i < 12; i++) {
        world_verts[i] = (Vector3){
            shape->vertices[i].x + shape->position.x,
            shape->vertices[i].y + shape->position.y,
            shape->vertices[i].z + shape->position.z
        };
    }

    // Draw all 20 solid faces (double-sided to avoid backface culling issues)
    for (int f = 0; f < 20; f++) {
        Vector3 v0 = world_verts[faces[f][0]];
        Vector3 v1 = world_verts[faces[f][1]];
        Vector3 v2 = world_verts[faces[f][2]];

        // Draw front face
        DrawTriangle3D(v0, v1, v2, color);
        // Draw back face (reversed winding)
        DrawTriangle3D(v0, v2, v1, color);
    }

    // Draw all 30 edges on top for visibility
    for (int e = 0; e < 30; e++) {
        Vector3 p0 = world_verts[edges[e][0]];
        Vector3 p1 = world_verts[edges[e][1]];
        DrawLine3D(p0, p1, BLACK);
    }
}

// Auto-initialize objects in a configurable pattern
void auto_initialize_objects(PhysicsObject* objects, int num_objects) {
    // Predefined colors for visual distinction
    Color colors[] = {RED, BLUE, GREEN, ORANGE, PURPLE, PINK, LIME, SKYBLUE, MAROON};
    int num_colors = sizeof(colors) / sizeof(colors[0]);

    // Auto-generate objects in a grid pattern above the ground
    int grid_size = (int)ceil(sqrt((double)num_objects));
    float spacing = 3.0f;  // Space between objects
    float start_height = 10.0f;

    for (int i = 0; i < num_objects; i++) {
        int row = i / grid_size;
        int col = i % grid_size;

        // Center the grid
        float x_offset = -(grid_size - 1) * spacing / 2.0f;
        float z_offset = -(grid_size - 1) * spacing / 2.0f;

        // Add small random noise to avoid exact overlaps (range: -0.2 to +0.2)
        float noise_scale = 0.2f;
        float pos_noise_x = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float pos_noise_y = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float pos_noise_z = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;

        float vel_noise_x = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float vel_noise_y = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float vel_noise_z = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;

        char name[32];
        sprintf(name, "Obj %d", i);

        objects[i] = (PhysicsObject){
            .position = {
                x_offset + col * spacing + pos_noise_x,
                start_height + (i % 3) * 2.0f + pos_noise_y,
                z_offset + row * spacing + pos_noise_z
            },
            .velocity = {
                ((i % 3) - 1) * 1.0f + vel_noise_x,
                vel_noise_y,
                ((i % 2) - 0.5f) * 2.0f + vel_noise_z
            },
            .acceleration = { 0.0f, -9.8f, 0.0f },  // Gravity
            .radius = 1.5f,
            .mass = 1.0f,
            .color = colors[i % num_colors]
        };
        strcpy(objects[i].name, name);
    }
}

// Camera controls with mouse rotation
void UpdateCameraCustom(Camera3D* camera, float boundary) {
    // Calculate current position relative to target
    float dx = camera->position.x - camera->target.x;
    float dy = camera->position.y - camera->target.y;
    float dz = camera->position.z - camera->target.z;

    // Mouse rotation (LEFT mouse button) - orbit around target
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouseDelta = GetMouseDelta();

        // Horizontal rotation (Y-axis)
        float angleH = mouseDelta.x * 0.003f;
        float cosH = cosf(-angleH);
        float sinH = sinf(-angleH);
        float newX = dx * cosH - dz * sinH;
        float newZ = dx * sinH + dz * cosH;
        dx = newX;
        dz = newZ;

        // Vertical rotation (simple pitch with clamping)
        float angleV = mouseDelta.y * 0.003f;
        float radius = sqrtf(dx*dx + dy*dy + dz*dz);
        float currentPitch = asinf(dy / radius);
        float newPitch = currentPitch + angleV;

        // Clamp pitch to avoid gimbal lock
        if (newPitch > -1.5f && newPitch < 1.5f) {
            float horizontalDist = sqrtf(dx*dx + dz*dz);
            dy = radius * sinf(newPitch);
            float newHorizontalDist = radius * cosf(newPitch);
            float scale = newHorizontalDist / horizontalDist;
            dx *= scale;
            dz *= scale;
        }
    }

    // Arrow key rotation
    float rotSpeed = 0.02f;

    // Left/Right arrows - horizontal rotation
    if (IsKeyDown(KEY_LEFT)) {
        float angleH = rotSpeed;
        float cosH = cosf(-angleH);
        float sinH = sinf(-angleH);
        float newX = dx * cosH - dz * sinH;
        float newZ = dx * sinH + dz * cosH;
        dx = newX;
        dz = newZ;
    }
    if (IsKeyDown(KEY_RIGHT)) {
        float angleH = -rotSpeed;
        float cosH = cosf(-angleH);
        float sinH = sinf(-angleH);
        float newX = dx * cosH - dz * sinH;
        float newZ = dx * sinH + dz * cosH;
        dx = newX;
        dz = newZ;
    }

    // Up/Down arrows - vertical rotation
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_DOWN)) {
        float angleV = IsKeyDown(KEY_UP) ? -rotSpeed : rotSpeed;
        float radius = sqrtf(dx*dx + dy*dy + dz*dz);
        float currentPitch = asinf(dy / radius);
        float newPitch = currentPitch + angleV;

        // Clamp pitch to avoid gimbal lock
        if (newPitch > -1.5f && newPitch < 1.5f) {
            float horizontalDist = sqrtf(dx*dx + dz*dz);
            dy = radius * sinf(newPitch);
            float newHorizontalDist = radius * cosf(newPitch);
            float scale = newHorizontalDist / horizontalDist;
            dx *= scale;
            dz *= scale;
        }
    }

    // Update camera position
    camera->position.x = camera->target.x + dx;
    camera->position.y = camera->target.y + dy;
    camera->position.z = camera->target.z + dz;

    // Keyboard movement (simple WASD controls)
    float moveSpeed = 0.5f;
    if (IsKeyDown(KEY_W)) camera->position.z -= moveSpeed;
    if (IsKeyDown(KEY_S)) camera->position.z += moveSpeed;
    if (IsKeyDown(KEY_A)) camera->position.x -= moveSpeed;
    if (IsKeyDown(KEY_D)) camera->position.x += moveSpeed;
    if (IsKeyDown(KEY_Q)) camera->position.y -= moveSpeed;
    if (IsKeyDown(KEY_E)) camera->position.y += moveSpeed;

    // Reset camera (scales with boundary size)
    if (IsKeyPressed(KEY_R)) {
        camera->position = (Vector3){ 0.0f, boundary * 0.8f, boundary * 1.6f };
        camera->target = (Vector3){ 0.0f, -2.0f, 0.0f };  // Look slightly down
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
    camera.target = (Vector3){ 0.0f, -2.0f, 0.0f };  // Look slightly down
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    
    // Create physics objects arrays
    PhysicsObject objects[NUM_OBJECTS];
    GJK_Shape gjk_shapes[NUM_OBJECTS];
    PhysicsObject initial_objects[NUM_OBJECTS]; // For reset functionality

    // Auto-initialize all objects
    auto_initialize_objects(objects, NUM_OBJECTS);

    // Create GJK shapes (icosahedron approximation of spheres)
    for (int i = 0; i < NUM_OBJECTS; i++) {
        gjk_shapes[i] = create_sphere_shape(objects[i].position, objects[i].radius);
    }

    // Store initial state for reset
    for (int i = 0; i < NUM_OBJECTS; i++) {
        initial_objects[i] = objects[i];
    }
    
    // Physics simulation variables (using macros from sim_config.h)
    float deltaTime = DELTA_TIME;

    // GPU initialization
    bool gpu_available = false;
    double gpu_time = 0;
    int num_pairs_generated = 0;  // Will be updated each frame by dynamic pair generation

#ifdef USE_CUDA
    GPU_GJK_Context* gpu_context = NULL;
    gpu_available = gpu_gjk_init(&gpu_context, NUM_OBJECTS, MAX_PAIRS);

    if (gpu_available) {
        // Register all objects with GPU
        for (int i = 0; i < NUM_OBJECTS; i++) {
            gpu_gjk_register_object(gpu_context, i, &gjk_shapes[i],
                                   objects[i].position, objects[i].velocity,
                                   objects[i].mass, objects[i].radius);
        }

        // Sync registered objects to GPU device memory
        gpu_gjk_sync_objects_to_device(gpu_context);

        // NOTE: Collision pairs will be generated dynamically each frame using spatial grid
        // No need to call gpu_gjk_set_collision_pairs() anymore

        printf("GPU Physics Simulation Initialized with Dynamic Broad-Phase Culling!\n");
    } else {
        printf("GPU not available\n");
        return 1;
    }
#endif

    // GPU physics parameters (using macros from sim_config.h)
    GPU_PhysicsParams gpu_params;
    gpu_params.gravity = (Vector3f){0.0f, GRAVITY_Y, 0.0f};
    gpu_params.deltaTime = DELTA_TIME;
    gpu_params.dampingCoeff = DAMPING_COEFF;
    gpu_params.boundarySize = COMPUTE_BOUNDARY(NUM_OBJECTS);
    gpu_params.collisionEpsilon = COLLISION_EPSILON;

    // Update camera position to scale with boundary size
    float boundary = gpu_params.boundarySize;
    camera.position = (Vector3){ 0.0f, boundary * 0.8f, boundary * 1.6f };
    camera.target = (Vector3){ 0.0f, -2.0f, 0.0f };  // Look slightly down

    SetTargetFPS(60);
    
    // Main game loop
    while (!WindowShouldClose()) {
#ifdef USE_CUDA
        clock_t start = clock();

        // Step 1: Dynamically generate collision pairs using spatial grid broad-phase culling
        num_pairs_generated = gpu_gjk_update_collision_pairs_dynamic(gpu_context, &gpu_params);

        // Step 2: Run full physics simulation on GPU (uses dynamically generated pairs)
        gpu_gjk_step_simulation(gpu_context, &gpu_params);

        clock_t end = clock();
        gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

        // Get render data from GPU
        GPU_RenderData render_data;
        gpu_gjk_get_render_data(gpu_context, &render_data);
#endif

        // Reset simulation
        if (IsKeyPressed(KEY_SPACE)) {
#ifdef USE_CUDA
            gpu_gjk_reset_simulation(gpu_context);
#endif
            printf("Simulation Reset!\n");
        }

        // Update camera (pass boundary for proper reset scaling)
        UpdateCameraCustom(&camera, gpu_params.boundarySize);

        // Drawing
        BeginDrawing();
            ClearBackground(RAYWHITE);

            BeginMode3D(camera);

            // Draw floor and ceiling (scales with boundary size)
            float groundSize = gpu_params.boundarySize * 2.0f;
            float groundY = -gpu_params.boundarySize;  // Ground at bottom of boundary box
            float ceilingY = gpu_params.boundarySize;   // Ceiling at top of boundary box

            DrawPlane((Vector3){0, groundY, 0}, (Vector2){groundSize, groundSize}, LIGHTGRAY);

#ifdef USE_CUDA
            // Draw all objects using GPU render data
            for (int i = 0; i < render_data.num_objects; i++) {
                // Update shape position for rendering
                gjk_shapes[i].position = render_data.positions[i];

                // Draw the solid polytope (icosahedron) with original color
                draw_polytope_solid(&gjk_shapes[i], objects[i].color);
            }
#endif
            
            // // Draw coordinate axes
            // DrawLine3D((Vector3){0,0,0}, (Vector3){5,0,0}, RED);   // X axis
            // DrawLine3D((Vector3){0,0,0}, (Vector3){0,5,0}, GREEN); // Y axis
            // DrawLine3D((Vector3){0,0,0}, (Vector3){0,0,5}, BLUE);  // Z axis
            
            EndMode3D();
            
            // Draw UI
            char title[128];
            sprintf(title, "Physics Simulation - %d Objects - GPU GJK", NUM_OBJECTS);
            DrawText(title, 10, 10, 20, DARKGRAY);
            
            // GPU status and timing
#ifdef USE_CUDA
            DrawText("GPU PHYSICS + BROAD-PHASE CULLING", screenWidth - 320, 10, 20, GREEN);
            char timing_text[128];
            sprintf(timing_text, "GPU Time: %.3f ms", gpu_time);
            DrawText(timing_text, 10, 40, 18, BLUE);

            char culling_text[128];
            sprintf(culling_text, "Pairs: %d / %d (%.1f%% culled)",
                    num_pairs_generated, MAX_PAIRS,
                    100.0f * (1.0f - (float)num_pairs_generated / MAX_PAIRS));
            DrawText(culling_text, 10, 60, 18, DARKGREEN);
#endif

            
        EndDrawing();
    }
    
    // Cleanup
#ifdef USE_CUDA
    if (gpu_available) {
        gpu_gjk_cleanup(&gpu_context);
    }
#endif
    for (int i = 0; i < NUM_OBJECTS; i++) {
        free_shape(&gjk_shapes[i]);
    }
    CloseWindow();

    return 0;
}