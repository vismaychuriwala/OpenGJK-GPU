#version 460 core

in vec3 v_world_normal;
in vec3 v_world_pos;
in vec4 v_color;

uniform vec3 uLightDir;
uniform vec3 uCameraPos;

out vec4 frag_color;

void main() {
    vec3 n = normalize(v_world_normal);
    vec3 l = normalize(uLightDir);
    vec3 v = normalize(uCameraPos - v_world_pos);
    vec3 h = normalize(l + v);

    // Hemisphere ambient: warm sky above, cool ground below
    float hemi = n.y * 0.5 + 0.5;
    vec3 ambient = mix(vec3(0.05, 0.06, 0.12), vec3(0.20, 0.18, 0.14), hemi);

    // Key light diffuse
    float diff_key = max(dot(n, l), 0.0);

    // Fill light: opposite side, cooler, dimmer
    vec3 l_fill = normalize(vec3(-0.6, 0.3, -0.4));
    float diff_fill = max(dot(n, l_fill), 0.0) * 0.25;

    // Blinn-Phong specular on key light (white highlight)
    float spec = pow(max(dot(n, h), 0.0), 48.0) * 0.35;

    // Rim: brightens silhouette edges toward camera
    float rim = pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 3.0) * 0.3;

    vec3 color = v_color.rgb * (ambient + vec3(diff_key * 0.8 + diff_fill))
               + vec3(spec)
               + v_color.rgb * rim;

    frag_color = vec4(color, v_color.a);
}
