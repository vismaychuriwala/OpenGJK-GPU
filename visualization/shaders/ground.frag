#version 330 core

in vec3 vWorldPos;
out vec4 FragColor;

uniform samplerCube u_DiffuseIrradianceMap;

void main() {
    // Lambertian ground lit by the baked diffuse irradiance (normal = +Y)
    vec3 groundAlbedo = vec3(0.45);
    vec3 irradiance = texture(u_DiffuseIrradianceMap, vec3(0.0, 1.0, 0.0)).rgb;
    vec3 color = irradiance * groundAlbedo;

    // Reinhard op + gamma correction (matches object/sky shaders)
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    FragColor = vec4(color, 1.0);
}