#version 460 core

in vec3 v_world_normal;
in vec3 v_world_pos;
in vec4 v_color;

uniform vec3 uLightDir;

out vec4 frag_color;

void main() {
    vec3 n = normalize(v_world_normal);
    vec3 l = normalize(uLightDir);

    float diffuse  = max(dot(n, l), 0.0);
    float ambient  = 0.15;
    float lighting = ambient + diffuse * 0.85;

    frag_color = vec4(v_color.rgb * lighting, v_color.a);
}
