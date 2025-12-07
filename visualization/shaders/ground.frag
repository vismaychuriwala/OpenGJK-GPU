#version 330 core

in vec3 vWorldPos;
out vec4 FragColor;

void main() {
    // Simple grid pattern
    float gridSize = 2.0;
    float x = abs(fract(vWorldPos.x / gridSize - 0.5) - 0.5);
    float z = abs(fract(vWorldPos.z / gridSize - 0.5) - 0.5);
    float grid = min(x, z);

    // Lighter gray base, darker grid lines
    vec3 baseColor = vec3(0.9);
    vec3 lineColor = vec3(0.7);
    float t = smoothstep(0.0, 0.05, grid);

    FragColor = vec4(mix(lineColor, baseColor, t), 1.0);
}
