#include "raylib.h"
#include "gjk_integration.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Include GPU headers at the TOP, not inside functions
#ifdef USE_CUDA
#include "gpu_gjk_interface.h"
#endif

// Configuration
// Change NUM_OBJECTS to test different scenarios:
// - NUM_OBJECTS=2  → 1 pair    (original setup, backwards compatible)
// - NUM_OBJECTS=3  → 3 pairs   (small triangle)
// - NUM_OBJECTS=5  → 10 pairs  (small grid)
// - NUM_OBJECTS=10 → 45 pairs  (medium grid)
// - NUM_OBJECTS=20 → 190 pairs (stress test)
// - NUM_OBJECTS=50 → 1225 pairs (large benchmark)
#define NUM_OBJECTS 10           // Number of physics objects
#define MAX_PAIRS ((NUM_OBJECTS * (NUM_OBJECTS - 1)) / 2)  // Compile-time constant for collision pairs

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

        char name[32];
        sprintf(name, "Obj %d", i);

        objects[i] = (PhysicsObject){
            .position = {
                x_offset + col * spacing,
                start_height + (i % 3) * 2.0f,  // Vary height slightly
                z_offset + row * spacing
            },
            .velocity = {
                ((i % 3) - 1) * 1.0f,  // Vary X velocity: -1, 0, 1
                0.0f,
                ((i % 2) - 0.5f) * 2.0f  // Vary Z velocity: -1 or 1
            },
            .acceleration = { 0.0f, -9.8f, 0.0f },  // Gravity
            .radius = 1.5f,
            .mass = 1.0f,
            .color = colors[i % num_colors]
        };
        strcpy(objects[i].name, name);
    }
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
    
    // Physics simulation variables
    float deltaTime = 1.0f / 60.0f; // Fixed timestep for stable physics

    // Collision detection arrays
    const int num_pairs = MAX_PAIRS;
    int collision_pairs[MAX_PAIRS * 2];
    bool collision_results[MAX_PAIRS];

    // Generate all unique collision pairs
    int pair_idx = 0;
    for (int i = 0; i < NUM_OBJECTS; i++) {
        for (int j = i + 1; j < NUM_OBJECTS; j++) {
            collision_pairs[pair_idx * 2 + 0] = i;
            collision_pairs[pair_idx * 2 + 1] = j;
            pair_idx++;
        }
    }
    
    // GPU support variables
    bool gpu_available = false;
    GJKMode current_mode = MODE_GPU; // Start with GPU mode
    double cpu_time = 0, gpu_time = 0;
    
    // Try to initialize GPU if available
#ifdef USE_CUDA
    GPU_GJK_Context* gpu_context = NULL;
    gpu_available = gpu_gjk_init(&gpu_context, NUM_OBJECTS);

    if (gpu_available) {
        // Register all shapes at startup
        for (int i = 0; i < NUM_OBJECTS; i++) {
            gpu_gjk_register_shape(gpu_context, &gjk_shapes[i], i);
        }
        current_mode = MODE_GPU;
        printf("GPU Physics Simulation Initialized!\n");
    } else {
        printf("GPU not available, using CPU physics\n");
    }
#endif
    
    SetTargetFPS(60);
    
    // Main game loop
    while (!WindowShouldClose()) {
        // Physics Update - loop over all objects
        for (int i = 0; i < NUM_OBJECTS; i++) {
            // Update velocity with acceleration (gravity)
            objects[i].velocity.x += objects[i].acceleration.x * deltaTime;
            objects[i].velocity.y += objects[i].acceleration.y * deltaTime;
            objects[i].velocity.z += objects[i].acceleration.z * deltaTime;

            // Update position with velocity
            objects[i].position.x += objects[i].velocity.x * deltaTime;
            objects[i].position.y += objects[i].velocity.y * deltaTime;
            objects[i].position.z += objects[i].velocity.z * deltaTime;

            // Ground collision
            if (objects[i].position.y - objects[i].radius < 0) {
                objects[i].position.y = objects[i].radius;
                objects[i].velocity.y = -objects[i].velocity.y * 0.8f; // Bounce with damping
            }

            // Wall collisions (simple boundary)
            float boundary = 15.0f;

            // X-axis boundaries
            if (fabs(objects[i].position.x) > boundary - objects[i].radius) {
                objects[i].velocity.x = -objects[i].velocity.x * 0.8f;
                objects[i].position.x = (objects[i].position.x > 0) ?
                    boundary - objects[i].radius : -boundary + objects[i].radius;
            }

            // Z-axis boundaries
            if (fabs(objects[i].position.z) > boundary - objects[i].radius) {
                objects[i].velocity.z = -objects[i].velocity.z * 0.8f;
                objects[i].position.z = (objects[i].position.z > 0) ?
                    boundary - objects[i].radius : -boundary + objects[i].radius;
            }

            // Update GJK shape position
            gjk_shapes[i].position = objects[i].position;
        }
        
        // Mode switching (only if GPU available)
        if (IsKeyPressed(KEY_TAB) && gpu_available) {
            current_mode = (current_mode == MODE_CPU) ? MODE_GPU : MODE_CPU;
            printf("Switched to %s physics\n", current_mode == MODE_GPU ? "GPU" : "CPU");
        }
        
        // Reset simulation
        if (IsKeyPressed(KEY_SPACE)) {
            for (int i = 0; i < NUM_OBJECTS; i++) {
                objects[i] = initial_objects[i];
            }
            printf("Simulation Reset!\n");
        }
        
        // Collision detection with timing
        clock_t start, end;

#ifdef USE_CUDA
        if (current_mode == MODE_GPU && gpu_available) {
            start = clock();

            // Update all positions on GPU
            for (int i = 0; i < NUM_OBJECTS; i++) {
                gpu_gjk_update_position(gpu_context, i, objects[i].position);
            }

            // Batch check all collision pairs on GPU
            gpu_gjk_batch_check(gpu_context, collision_pairs, num_pairs, collision_results);

            end = clock();
            gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
        } else {
#endif
            // CPU collision detection for all pairs
            start = clock();
            for (int p = 0; p < num_pairs; p++) {
                int idA = collision_pairs[p * 2 + 0];
                int idB = collision_pairs[p * 2 + 1];
                collision_results[p] = openGJK_collision_cpu(&gjk_shapes[idA], &gjk_shapes[idB]);
            }
            end = clock();
            cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
#ifdef USE_CUDA
        }
#endif
        
        // Handle collision response for all detected collisions
        for (int p = 0; p < num_pairs; p++) {
            if (collision_results[p]) {
                int idA = collision_pairs[p * 2 + 0];
                int idB = collision_pairs[p * 2 + 1];
                resolve_collision(&objects[idA], &objects[idB]);
            }
        }

        // Calculate distance for display (first pair only)
        float distance = 0.0f;
        if (num_pairs > 0) {
            int idA = collision_pairs[0];
            int idB = collision_pairs[1];
            float dx = objects[idA].position.x - objects[idB].position.x;
            float dy = objects[idA].position.y - objects[idB].position.y;
            float dz = objects[idA].position.z - objects[idB].position.z;
            distance = sqrt(dx*dx + dy*dy + dz*dz);
        }
        
        // Update camera
        UpdateCameraCustom(&camera);
        
        // Drawing
        BeginDrawing();
            ClearBackground(RAYWHITE);
            
            BeginMode3D(camera);
            
            // Draw floor
            DrawPlane((Vector3){0, 0, 0}, (Vector2){50, 50}, LIGHTGRAY);

            // Determine which objects are colliding
            bool object_colliding[NUM_OBJECTS];
            for (int i = 0; i < NUM_OBJECTS; i++) {
                object_colliding[i] = false;
            }
            for (int p = 0; p < num_pairs; p++) {
                if (collision_results[p]) {
                    object_colliding[collision_pairs[p * 2 + 0]] = true;
                    object_colliding[collision_pairs[p * 2 + 1]] = true;
                }
            }

            // Draw all objects as polytopes (what we actually test with GJK)
            for (int i = 0; i < NUM_OBJECTS; i++) {
                Color drawColor = object_colliding[i] ? YELLOW : objects[i].color;

                // Draw the solid polytope (icosahedron)
                draw_polytope_solid(&gjk_shapes[i], drawColor);
            }
            
            // Draw coordinate axes
            DrawLine3D((Vector3){0,0,0}, (Vector3){5,0,0}, RED);   // X axis
            DrawLine3D((Vector3){0,0,0}, (Vector3){0,5,0}, GREEN); // Y axis
            DrawLine3D((Vector3){0,0,0}, (Vector3){0,0,5}, BLUE);  // Z axis
            
            EndMode3D();
            
            // Draw UI
            char title[128];
            sprintf(title, "Physics Simulation - %d Objects - GPU GJK", NUM_OBJECTS);
            DrawText(title, 10, 10, 20, DARKGRAY);
            
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
            
            // Physics info - count total collisions
            int total_collisions = 0;
            for (int p = 0; p < num_pairs; p++) {
                if (collision_results[p]) total_collisions++;
            }

            if (total_collisions > 0) {
                char collision_text[64];
                sprintf(collision_text, "COLLISIONS: %d - BOUNCING", total_collisions);
                DrawText(collision_text, 10, 100, 30, RED);
            } else {
                DrawText("No collisions", 10, 100, 30, GREEN);
            }

            // Show collision pair count and distance
            char pairsText[64];
            sprintf(pairsText, "Checking %d collision pairs", num_pairs);
            DrawText(pairsText, 10, 140, 20, DARKGRAY);

            char distanceText[50];
            sprintf(distanceText, "Distance (pair 0): %.2f", distance);
            DrawText(distanceText, 10, 165, 18, DARKGRAY);

            // Show velocity info for first 2 objects (keeps UI simple)
            char velocityText[80];
            for (int i = 0; i < NUM_OBJECTS && i < 2; i++) {
                sprintf(velocityText, "%s Vel: (%.1f, %.1f, %.1f)",
                        objects[i].name,
                        objects[i].velocity.x, objects[i].velocity.y, objects[i].velocity.z);
                DrawText(velocityText, 10, 170 + i * 25, 18, objects[i].color);
            }
            
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
            
            // Draw coordinates for first 2 objects
            char posText[80];
            for (int i = 0; i < NUM_OBJECTS && i < 2; i++) {
                sprintf(posText, "%s: (%.1f, %.1f, %.1f)",
                        objects[i].name,
                        objects[i].position.x, objects[i].position.y, objects[i].position.z);
                DrawText(posText, 10, screenHeight - 60 + i * 25, 18, objects[i].color);
            }
            
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