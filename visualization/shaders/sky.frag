#version 330 core

in vec3 vDir;
uniform samplerCube u_EnvironmentMap;
out vec4 FragColor;

void main() {
    vec3 envColor = texture(u_EnvironmentMap, normalize(vDir)).rgb;

    // Reinhard op + gamma correction
    envColor = envColor / (envColor + vec3(1.0));
    envColor = pow(envColor, vec3(1.0/2.2));

    FragColor = vec4(envColor, 1.0);
}