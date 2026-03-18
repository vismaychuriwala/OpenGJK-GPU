#include "input.h"
#include <cstring>

static InputState g_input = {0};

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key < 0 || key >= GLFW_KEY_LAST) return;

    if (action == GLFW_PRESS) {
        g_input.keys[key] = true;
        g_input.keys_pressed[key] = true;
    } else if (action == GLFW_RELEASE) {
        g_input.keys[key] = false;
    }
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button < 0 || button >= GLFW_MOUSE_BUTTON_LAST) return;
    g_input.mouse_buttons[button] = (action == GLFW_PRESS);
}

static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    // Accumulate delta from each move event fired during glfwPollEvents
    g_input.mouse_delta_x += xpos - g_input.mouse_x;
    g_input.mouse_delta_y += ypos - g_input.mouse_y;
    g_input.mouse_x = xpos;
    g_input.mouse_y = ypos;
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    g_input.scroll_offset += yoffset;
}

void input_init(GLFWwindow* window) {
    std::memset(&g_input, 0, sizeof(InputState));

    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwGetCursorPos(window, &g_input.mouse_x, &g_input.mouse_y);
}

void input_update(GLFWwindow* window) {
    bool prev_lmb = g_input.mouse_buttons[GLFW_MOUSE_BUTTON_LEFT];

    // Reset per-frame accumulators before polling
    std::memset(g_input.keys_pressed, 0, sizeof(g_input.keys_pressed));
    g_input.scroll_offset = 0.0;
    g_input.mouse_delta_x = 0.0;
    g_input.mouse_delta_y = 0.0;

    // cursor_pos_callback accumulates delta for all move events this frame
    glfwPollEvents();

    // On the first frame of a drag, discard delta — it may include pre-click movement
    bool lmb_just_pressed = !prev_lmb && g_input.mouse_buttons[GLFW_MOUSE_BUTTON_LEFT];
    if (lmb_just_pressed) {
        g_input.mouse_delta_x = 0.0;
        g_input.mouse_delta_y = 0.0;
    }
}

// Raylib-compatible helper functions
bool IsKeyDown(int key) {
    if (key < 0 || key >= GLFW_KEY_LAST) return false;
    return g_input.keys[key];
}

bool IsKeyPressed(int key) {
    if (key < 0 || key >= GLFW_KEY_LAST) return false;
    return g_input.keys_pressed[key];
}

bool IsMouseButtonDown(int button) {
    if (button < 0 || button >= GLFW_MOUSE_BUTTON_LAST) return false;
    return g_input.mouse_buttons[button];
}

float GetMouseWheelMove(void) {
    return (float)g_input.scroll_offset;
}

void GetMouseDelta(float* dx, float* dy) {
    *dx = (float)g_input.mouse_delta_x;
    *dy = (float)g_input.mouse_delta_y;
}
