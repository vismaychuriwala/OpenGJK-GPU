#version 330 core
layout(location = 0) in vec3 vs_Pos;

out vec3 fs_Pos;

// Actually just a view matrix that
// rotates the cube by 90 degrees
// such that we can render it 6 times
// and have each face aligned with the
// NDC screen coords.
uniform mat4 u_ViewProj;

void main() {
    fs_Pos = vs_Pos;
    gl_Position =  u_ViewProj * vec4(fs_Pos, 1.0);
}