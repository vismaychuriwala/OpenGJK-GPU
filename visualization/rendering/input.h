#ifndef INPUT_H
#define INPUT_H

#include <GLFW/glfw3.h>

struct InputState {
    // Keyboard state
    bool keys[GLFW_KEY_LAST];
    bool keys_pressed[GLFW_KEY_LAST];  // One-shot events

    // Mouse state
    bool mouse_buttons[GLFW_MOUSE_BUTTON_LAST];
    double mouse_x, mouse_y;
    double mouse_delta_x, mouse_delta_y;
    double scroll_offset;

    // Previous frame state
    double prev_mouse_x, prev_mouse_y;
};

// Initialize input system and set up GLFW callbacks
void input_init(GLFWwindow* window);

// Call this every frame to update input state
void input_update(GLFWwindow* window);

// Raylib-compatible helper functions
bool IsKeyDown(int key);
bool IsKeyPressed(int key);
bool IsMouseButtonDown(int button);
float GetMouseWheelMove(void);
void GetMouseDelta(float* dx, float* dy);

#endif // INPUT_H
