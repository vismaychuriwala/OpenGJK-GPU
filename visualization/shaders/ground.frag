#version 330 core

in vec3 vWorldPos;
out vec4 FragColor;

void main() {
    // Solid grey ground
    vec3 groundColor = vec3(0.8, 0.8, 0.8);  // Light grey
    FragColor = vec4(groundColor, 1.0);
}
