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
#include "gpu_gjk_interface.h"

struct PhysicsObject {
    float position[3];
    float velocity[3];
    float radius;
    float mass;
    float color[4];
};

void auto_initialize_objects(PhysicsObject* objects, int num_objects) {
    float colors[][4] = {
        {0.9f, 0.2f, 0.2f, 1.0f},
        {0.2f, 0.2f, 0.9f, 1.0f},
        {0.2f, 0.9f, 0.2f, 1.0f},
        {0.9f, 0.6f, 0.2f, 1.0f},
        {0.7f, 0.2f, 0.9f, 1.0f},
        {0.9f, 0.5f, 0.8f, 1.0f},
        {0.5f, 0.9f, 0.2f, 1.0f},
        {0.4f, 0.7f, 0.9f, 1.0f},
        {0.6f, 0.2f, 0.3f, 1.0f}
    };
    int num_colors = sizeof(colors) / sizeof(colors[0]);

    int grid_size = (int)std::ceil(std::sqrt((double)num_objects));
    float spacing = 3.0f;
    float start_height = 10.0f;

    for (int i = 0; i < num_objects; i++) {
        int row = i / grid_size;
        int col = i % grid_size;

        float x_offset = -(grid_size - 1) * spacing / 2.0f;
        float z_offset = -(grid_size - 1) * spacing / 2.0f;

        float noise_scale = 0.2f;
        float pos_noise_x = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float pos_noise_y = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float pos_noise_z = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;

        float vel_noise_x = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float vel_noise_y = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;
        float vel_noise_z = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * noise_scale;

        float radius = 0.5f + ((float)rand() / RAND_MAX) * 1.3f;

        objects[i].position[0] = x_offset + col * spacing + pos_noise_x;
        objects[i].position[1] = start_height + (i % 3) * 2.0f + pos_noise_y;
        objects[i].position[2] = z_offset + row * spacing + pos_noise_z;

        objects[i].velocity[0] = ((i % 3) - 1) * 1.0f + vel_noise_x;
        objects[i].velocity[1] = vel_noise_y;
        objects[i].velocity[2] = ((i % 2) - 0.5f) * 2.0f + vel_noise_z;

        objects[i].radius = radius;
        objects[i].mass = radius * radius * radius;

        std::memcpy(objects[i].color, colors[i % num_colors], sizeof(float) * 4);
    }
}

int main(void) {
    const int screenWidth = 1200;
    const int screenHeight = 800;

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight,
                                          "Physics Simulation - OpenGL", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.95f, 0.95f, 0.95f, 1.0f);

    input_init(window);

    Camera3D camera;
    float boundary = COMPUTE_BOUNDARY(NUM_OBJECTS);
    glm::vec3 cam_pos(0.0f, boundary * 0.8f, boundary * 1.6f);
    glm::vec3 cam_target(0.0f, -2.0f, 0.0f);
    camera_init(&camera, cam_pos, cam_target, 45.0f);

    OpenGLRenderer renderer;
    if (!renderer_init(&renderer, NUM_OBJECTS)) {
        fprintf(stderr, "Failed to initialize renderer\n");
        return -1;
    }

    std::srand((unsigned int)std::time(nullptr));
    PhysicsObject* objects = new PhysicsObject[NUM_OBJECTS];
    auto_initialize_objects(objects, NUM_OBJECTS);

    GPU_GJK_Context* gpu_ctx = nullptr;
    if (!gpu_gjk_init(&gpu_ctx, NUM_OBJECTS, MAX_PAIRS)) {
        fprintf(stderr, "GPU initialization failed!\n");
        return -1;
    }

    GJK_Shape* gjk_shapes = new GJK_Shape[NUM_OBJECTS];
    for (int i = 0; i < NUM_OBJECTS; i++) {
        Vector3f pos = { objects[i].position[0], objects[i].position[1], objects[i].position[2] };
        gjk_shapes[i] = create_sphere_shape(pos, objects[i].radius);
    }

    for (int i = 0; i < NUM_OBJECTS; i++) {
        Vector3f pos = { objects[i].position[0], objects[i].position[1], objects[i].position[2] };
        Vector3f vel = { objects[i].velocity[0], objects[i].velocity[1], objects[i].velocity[2] };
        gpu_gjk_register_object(gpu_ctx, i, &gjk_shapes[i], pos, vel,
                                objects[i].mass, objects[i].radius);
    }

    for (int i = 0; i < NUM_OBJECTS; i++) {
        free_shape(&gjk_shapes[i]);
    }
    delete[] gjk_shapes;

    gpu_gjk_sync_objects_to_device(gpu_ctx);

    GPU_PhysicsParams params;
    params.gravity = { 0.0f, GRAVITY_Y, 0.0f };
    params.deltaTime = DELTA_TIME;
    params.dampingCoeff = DAMPING_COEFF;
    params.boundarySize = boundary;
    params.collisionEpsilon = COLLISION_EPSILON;

    double last_time = glfwGetTime();
    double last_physics_time = last_time;
    double last_frame_time = last_time;
    int frame_count = 0;
    int fps = 0;
    int collision_count = 0;
    double physics_accumulator = 0.0;

    while (!glfwWindowShouldClose(window)) {
        frame_count++;
        double current_time = glfwGetTime();
        float deltaTime = (float)(current_time - last_frame_time);
        last_frame_time = current_time;

        if (current_time - last_time >= 1.0) {
            fps = frame_count;
            frame_count = 0;
            last_time = current_time;
        }

        input_update(window);

        if (IsKeyPressed(GLFW_KEY_ESCAPE)) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (IsKeyPressed(GLFW_KEY_R)) {
            camera_reset(&camera);
        }
        if (IsKeyPressed(GLFW_KEY_F)) {
            renderer_toggle_wireframe(&renderer);
        }

        camera_update_controls(&camera, deltaTime);

        float aspect = (float)screenWidth / (float)screenHeight;
        camera_update_matrices(&camera, aspect);

        double current_physics_time = glfwGetTime();
        physics_accumulator += (current_physics_time - last_physics_time);
        last_physics_time = current_physics_time;

        if (physics_accumulator > 0.2) physics_accumulator = 0.2;

        while (physics_accumulator >= DELTA_TIME) {
            collision_count = gpu_gjk_update_collision_pairs_dynamic(gpu_ctx, &params);
            gpu_gjk_step_simulation(gpu_ctx, &params);
            physics_accumulator -= DELTA_TIME;
        }

        GPU_RenderData render_data;
        gpu_gjk_get_render_data(gpu_ctx, &render_data);

        float* positions = (float*)malloc(NUM_OBJECTS * 3 * sizeof(float));
        float* radii     = (float*)malloc(NUM_OBJECTS * sizeof(float));
        float* colors    = (float*)malloc(NUM_OBJECTS * 4 * sizeof(float));

        for (int i = 0; i < NUM_OBJECTS; i++) {
            positions[i * 3 + 0] = render_data.positions[i].x;
            positions[i * 3 + 1] = render_data.positions[i].y;
            positions[i * 3 + 2] = render_data.positions[i].z;
            radii[i] = objects[i].radius;
            std::memcpy(&colors[i * 4], objects[i].color, sizeof(float) * 4);
        }

        renderer_update_instances(&renderer, positions, radii, colors, NUM_OBJECTS);

        free(positions);
        free(radii);
        free(colors);

        renderer_draw(&renderer, camera.projection_matrix, camera.view_matrix);

        static int stats_counter = 0;
        if (++stats_counter % 60 == 0) {
            std::printf("FPS: %d | Pairs: %d | Objects: %d\n",
                        fps, collision_count, NUM_OBJECTS);
        }

        glfwSwapBuffers(window);
    }

    renderer_cleanup(&renderer);
    delete[] objects;
    gpu_gjk_cleanup(&gpu_ctx);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
