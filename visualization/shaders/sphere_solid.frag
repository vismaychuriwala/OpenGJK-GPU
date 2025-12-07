#version 330 core

in vec3 vNormal;
in vec4 vColor;
in vec3 vWorldPos;

out vec4 FragColor;

uniform vec3 uLightDir = vec3(0.5, 0.7, 0.3);

void main() {
    // Simple diffuse lighting
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightDir);

    float diffuse = max(dot(normal, lightDir), 0.0);

    // Add ambient light so objects aren't completely black
    float ambient = 0.3;
    float lighting = ambient + diffuse * 0.7;

    FragColor = vec4(vColor.rgb * lighting, vColor.a);
}
