#version 460 core

in vec3 v_world_normal;
in vec3 v_world_pos;
in vec4 v_color;
in vec2 v_uv;
flat in float v_tex_index;

uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform sampler2D uEnvMap;
uniform bool uHasEnvMap;
uniform sampler2DArray uTexArray;

out vec4 frag_color;

vec2 sampleSphericalMap(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= vec2(0.1591, 0.3183);
    return uv + 0.5;
}

void main() {
    vec3 n = normalize(v_world_normal);
    vec3 l = normalize(uLightDir);
    vec3 v = normalize(uCameraPos - v_world_pos);
    vec3 h = normalize(l + v);

    // Base color: texture array layer or flat object color
    vec3 base_color;
    if (v_tex_index >= 0.0) {
        base_color = texture(uTexArray, vec3(v_uv, v_tex_index)).rgb;
    } else {
        base_color = v_color.rgb;
    }

    // Ambient: env map at blurred mip or hemisphere fallback
    vec3 ambient;
    if (uHasEnvMap) {
        vec3 env = textureLod(uEnvMap, sampleSphericalMap(n), 5.0).rgb;
        env = env / (env + vec3(1.0));
        ambient = env * 0.8;
    } else {
        float hemi = n.y * 0.5 + 0.5;
        ambient = mix(vec3(0.05, 0.06, 0.12), vec3(0.20, 0.18, 0.14), hemi);
    }

    // Key light diffuse
    float diff_key = max(dot(n, l), 0.0);

    // Fill light: opposite side, cooler, dimmer
    vec3 l_fill = normalize(vec3(-0.6, 0.3, -0.4));
    float diff_fill = max(dot(n, l_fill), 0.0) * 0.25;

    // Blinn-Phong specular on key light
    float spec = pow(max(dot(n, h), 0.0), 48.0) * 0.35;

    // Rim: brightens silhouette edges toward camera
    float rim = pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 3.0) * 0.3;

    vec3 color = base_color * (ambient + vec3(diff_key * 0.8 + diff_fill))
               + vec3(spec)
               + base_color * rim;

    frag_color = vec4(color, v_color.a);
}
