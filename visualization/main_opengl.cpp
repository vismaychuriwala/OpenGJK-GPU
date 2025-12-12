#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm.hpp>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>

#include "rendering/input.h"
#include "rendering/camera.h"
#include "rendering/opengl_renderer.h"
#include "sim_config.h"
#include "gjk_integration.h"

#ifdef USE_CUDA
#include "gpu_gjk_interface.h"
#endif

// Physics Object structure (matching main.c)
struct PhysicsObject {
    float position[3];  // vec3
    float velocity[3];
    float acceleration[3];
    float radius;
    float mass;
    float color[4];  // RGBA
    char name[32];
};

// GPU GJK context and mode
enum class GJKMode { CPU, GPU };

// Auto-initialize objects in a grid pattern
void auto_initialize_objects(PhysicsObject* objects, int num_objects) {
    // Predefined colors for visual distinction
    float colors[][4] = {
        {0.9f, 0.2f, 0.2f, 1.0f},  // RED
        {0.2f, 0.2f, 0.9f, 1.0f},  // BLUE
        {0.2f, 0.9f, 0.2f, 1.0f},  // GREEN
        {0.9f, 0.6f, 0.2f, 1.0f},  // ORANGE
        {0.7f, 0.2f, 0.9f, 1.0f},  // PURPLE
        {0.9f, 0.5f, 0.8f, 1.0f},  // PINK
        {0.5f, 0.9f, 0.2f, 1.0f},  // LIME
        {0.4f, 0.7f, 0.9f, 1.0f},  // SKYBLUE
        {0.6f, 0.2f, 0.3f, 1.0f}   // MAROON
    };
    int num_colors = sizeof(colors) / sizeof(colors[0]);

    // Auto-generate objects in a grid pattern above the ground
    int grid_size = (int)std::ceil(std::sqrt((double)num_objects));
    float spacing = 3.0f;
    float start_height = 10.0f;

    for (int i = 0; i < num_objects; i++) {
        int row = i / grid_size;
        int col = i % grid_size;

        // Center the grid
        float x_offset = -(grid_size - 1) * spacing / 2.0f;
        float z_offset = -(grid_size - 1) * spacing / 2.0f;

        // Add small random noise to avoid exact overlaps
        float noise_scale = 0.2f;
        float pos_noise_x = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float pos_noise_y = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float pos_noise_z = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;

        float vel_noise_x = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float vel_noise_y = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float vel_noise_z = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;

        // Random varied sizes: range from 0.5 to 1.8
        float radius = 0.5f + ((float)rand() / RAND_MAX) * 1.3f;

        // Mass proportional to volume (radius^3) for realistic physics
        float mass = radius * radius * radius;

        objects[i].position[0] = x_offset + col * spacing + pos_noise_x;
        objects[i].position[1] = start_height + (i % 3) * 2.0f + pos_noise_y;
        objects[i].position[2] = z_offset + row * spacing + pos_noise_z;

        objects[i].velocity[0] = ((i % 3) - 1) * 1.0f + vel_noise_x;
        objects[i].velocity[1] = vel_noise_y;
        objects[i].velocity[2] = ((i % 2) - 0.5f) * 2.0f + vel_noise_z;

        objects[i].acceleration[0] = 0.0f;
        objects[i].acceleration[1] = -9.8f;  // Gravity
        objects[i].acceleration[2] = 0.0f;

        objects[i].radius = radius;
        objects[i].mass = mass;

        // Copy color
        std::memcpy(objects[i].color, colors[i % num_colors], sizeof(float) * 4);

        std::sprintf(objects[i].name, "Obj %d", i);
    }
}

int main(void) {
    const int screenWidth = 1200;
    const int screenHeight = 800;

    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    // Configure GLFW (OpenGL 3.3 Core)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // MSAA

    // Create window
    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight,
                                          "Physics Simulation - OpenGL", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable VSync

    // Load OpenGL functions with GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return -1;
    }

    // Print OpenGL version
    std::printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    std::printf("GLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    // Enable OpenGL features
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // Light gray background (RAYWHITE)

    // Initialize input system
    input_init(window);

    // Initialize camera
    Camera3D camera;
    float boundary = COMPUTE_BOUNDARY(NUM_OBJECTS);
    glm::vec3 cam_pos(0.0f, boundary * 0.8f, boundary * 1.6f);
    glm::vec3 cam_target(0.0f, -2.0f, 0.0f);
    camera_init(&camera, cam_pos, cam_target, 45.0f);

    // Initialize renderer
    OpenGLRenderer renderer;
    if (!renderer_init(&renderer, NUM_OBJECTS)) {
        fprintf(stderr, "Failed to initialize renderer\n");
        return -1;
    }

    // Initialize physics objects
    std::srand((unsigned int)std::time(nullptr));
    PhysicsObject* objects = new PhysicsObject[NUM_OBJECTS];
    auto_initialize_objects(objects, NUM_OBJECTS);

    // Initialize GPU physics context
    #ifdef USE_CUDA
    GPU_GJK_Context* gpu_ctx = nullptr;
    GJKMode mode = GJKMode::GPU;

    std::printf("Initializing GPU physics with %d objects...\n", NUM_OBJECTS);
    if (!gpu_gjk_init(&gpu_ctx, NUM_OBJECTS, MAX_PAIRS)) {
        std::fprintf(stderr, "GPU initialization failed! Falling back to CPU mode.\n");
        mode = GJKMode::CPU;
    } else {
        std::printf("GPU context initialized successfully!\n");

        // Create GJK shapes (icosahedron approximation of spheres)
        GJK_Shape* gjk_shapes = new GJK_Shape[NUM_OBJECTS];
        for (int i = 0; i < NUM_OBJECTS; i++) {
            Vector3f pos;
            pos.x = objects[i].position[0];
            pos.y = objects[i].position[1];
            pos.z = objects[i].position[2];

            gjk_shapes[i] = create_sphere_shape(pos, objects[i].radius);
        }

        // Register all objects with GPU
        for (int i = 0; i < NUM_OBJECTS; i++) {
            Vector3f pos;
            pos.x = objects[i].position[0];
            pos.y = objects[i].position[1];
            pos.z = objects[i].position[2];

            Vector3f vel;
            vel.x = objects[i].velocity[0];
            vel.y = objects[i].velocity[1];
            vel.z = objects[i].velocity[2];

            gpu_gjk_register_object(gpu_ctx, i, &gjk_shapes[i], pos, vel,
                                   objects[i].mass, objects[i].radius);
        }

        // Clean up gjk_shapes after registration (GPU has copied the data)
        for (int i = 0; i < NUM_OBJECTS; i++) {
            free_shape(&gjk_shapes[i]);
        }

        gpu_gjk_sync_objects_to_device(gpu_ctx);
        std::printf("GPU objects synchronized!\n");
    }
    #else
    GJKMode mode = GJKMode::CPU;
    std::printf("CUDA not enabled. CPU mode only.\n");
    #endif

    // Physics parameters
    GPU_PhysicsParams params;
    params.gravity.x = 0.0f;
    params.gravity.y = GRAVITY_Y;
    params.gravity.z = 0.0f;
    params.deltaTime = DELTA_TIME;
    params.dampingCoeff = DAMPING_COEFF;
    params.boundarySize = boundary;
    params.collisionEpsilon = COLLISION_EPSILON;

    int frame_count = 0;
    double last_time = glfwGetTime();
    double last_physics_time = last_time;
    double last_frame_time = last_time;
    int fps = 0;
    int collision_count = 0;
    double physics_accumulator = 0.0;

    std::printf("\n=== SIMULATION INFO ===\n");
    std::printf("Boundary size: %.2f\n", boundary);
    std::printf("Ground Y position: %.2f\n", -boundary);
    std::printf("Delta time: %.4f (%.1f FPS target)\n", DELTA_TIME, 1.0f/DELTA_TIME);
    std::printf("Gravity: %.2f\n", GRAVITY_Y);
    std::printf("\n=== CONTROLS ===\n");
    std::printf("WASD/QE: Move camera (world coordinates)\n");
    std::printf("Arrow Keys: Rotate camera\n");
    std::printf("Left Mouse: Drag to rotate\n");
    std::printf("Mouse Scroll: Zoom\n");
    std::printf("R: Reset camera\n");
    std::printf("F: Toggle wireframe\n");
    std::printf("ESC: Exit\n\n");

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate FPS and delta time
        frame_count++;
        double current_time = glfwGetTime();
        float deltaTime = (float)(current_time - last_frame_time);
        last_frame_time = current_time;

        if (current_time - last_time >= 1.0) {
            fps = frame_count;
            frame_count = 0;
            last_time = current_time;
        }

        // Handle input
        input_update(window);

        // Check for ESC to exit
        if (IsKeyPressed(GLFW_KEY_ESCAPE)) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // Reset simulation removed to save memory (no initial state storage)

        // Reset camera
        if (IsKeyPressed(GLFW_KEY_R)) {
            camera_reset(&camera);
        }

        // Toggle wireframe
        if (IsKeyPressed(GLFW_KEY_F)) {
            renderer_toggle_wireframe(&renderer);
            std::printf("Wireframe: %s\n", renderer.wireframe_enabled ? "ON" : "OFF");
        }

        // Update camera controls with delta time for frame-rate independence
        camera_update_controls(&camera, deltaTime);

        // Update camera matrices
        float aspect = (float)screenWidth / (float)screenHeight;
        camera_update_matrices(&camera, aspect);

        // Physics step - Fixed timestep at exactly 60 FPS
        double current_physics_time = glfwGetTime();
        physics_accumulator += (current_physics_time - last_physics_time);
        last_physics_time = current_physics_time;

        // Cap accumulator to prevent spiral of death
        if (physics_accumulator > 0.2) physics_accumulator = 0.2;

        // Step physics at fixed 60 FPS rate (step only when enough time has accumulated)
        while (physics_accumulator >= DELTA_TIME) {
            #ifdef USE_CUDA
            if (mode == GJKMode::GPU) {
                // Update collision pairs and step simulation on GPU
                collision_count = gpu_gjk_update_collision_pairs_dynamic(gpu_ctx, &params);
                gpu_gjk_step_simulation(gpu_ctx, &params);
            }
            #endif

            physics_accumulator -= DELTA_TIME;
        }

        // Get render data after all physics steps (update visuals once per frame)
        #ifdef USE_CUDA
        if (mode == GJKMode::GPU) {
            GPU_RenderData render_data;
            gpu_gjk_get_render_data(gpu_ctx, &render_data);

            // Extract data for renderer
            float* positions = (float*)malloc(NUM_OBJECTS * 3 * sizeof(float));
            float* radii = (float*)malloc(NUM_OBJECTS * sizeof(float));
            float* colors = (float*)malloc(NUM_OBJECTS * 4 * sizeof(float));

            for (int i = 0; i < NUM_OBJECTS; i++) {
                positions[i * 3 + 0] = render_data.positions[i].x;
                positions[i * 3 + 1] = render_data.positions[i].y;
                positions[i * 3 + 2] = render_data.positions[i].z;
                radii[i] = objects[i].radius;
                std::memcpy(&colors[i * 4], objects[i].color, sizeof(float) * 4);
            }

            // Update renderer instance buffer
            renderer_update_instances(&renderer, positions, radii, colors, NUM_OBJECTS);

            free(positions);
            free(radii);
            free(colors);
        }
        #endif

        // Render
        renderer_draw(&renderer, camera.projection_matrix, camera.view_matrix);

        // Simple text rendering (OpenGL immediate mode for now, TODO: proper text)
        // For now, just print stats to console occasionally
        static int stats_counter = 0;
        if (++stats_counter % 60 == 0) {
            std::printf("FPS: %d | Pairs: %d | Objects: %d\n",
                   fps, collision_count, NUM_OBJECTS);
        }

        // Swap buffers
        glfwSwapBuffers(window);
    }

    // Cleanup
    std::printf("\nCleaning up...\n");
    renderer_cleanup(&renderer);
    delete[] objects;

    #ifdef USE_CUDA
    if (mode == GJKMode::GPU && gpu_ctx) {
        gpu_gjk_cleanup(&gpu_ctx);
    }
    #endif

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
