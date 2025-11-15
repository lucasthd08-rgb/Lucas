// main.cpp
// Pixelated black hole + fused photon ring + horizontal pixel accretion disk + blurred stars
// Maintains the same structure you provided, with adjustments:
//  - larger black hole (mostly black)
//  - thin photon ring placed just outside BH
//  - accretion disk made of dense pixels (horizontal) and color-blended with ring
//  - disk and ring visually fused with smooth transitions (orange tone)
//  - stars blurred as before
//
// Build: g++ main.cpp -lglfw -lGLEW -lGL -ldl -pthread -o blackhole
// Requires GLFW, GLEW, glm.

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cstddef>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
using namespace glm;
using namespace std;

// -------------------- Settings (tweak these) --------------------
const int WIN_W = 800;
const int WIN_H = 600;

// Pixel resolution for the black hole (higher -> denser pixels)
int BH_PIXEL_RES = 64;        // increased for denser black hole pixels
float BH_RADIUS = 0.5f;       // larger billboard radius units (scene units)
float pixelPointSize = 6.0f;   // size in screen pixels for each "block" (smaller = denser look)

// Disk pixel parameters (denser than before to avoid visible gaps)
int DISK_RADIAL_STEPS = 10;     // rings (increased)
int DISK_ANGULAR_STEPS = 120;   // angular samples per ring (increased)
float DISK_INNER = 0.52f;       // inner radius (scene units)
float DISK_OUTER = 1.05f;      // outer radius (scene units)
float DISK_THICKNESS = 0.0001f;  // half thickness in Y (scene units)

// Photon ring (slightly outside BH)
float PH_RING_IN = BH_RADIUS * 1.06f;
float PH_RING_OUT = BH_RADIUS * 1.12f;
int PHOTON_SAMPLES = 1000;     // more samples for smooth continuous ring

// Stars
struct Star { vec3 pos; vec3 color; float size; };
vector<Star> stars;

// Camera
struct Camera {
    float radius = 3.5f;
    float azimuth = 0.0f;
    float elevation = glm::pi<float>() / 2.0f;
    bool dragging = false;
    double lastX = 0.0, lastY = 0.0;
    float orbitSpeed = 0.005f;
    float zoomSpeed = 0.3f;
    vec3 target = vec3(0.0f);
    vec3 position() const {
        float ele = clamp(elevation, 0.01f, glm::pi<float>() - 0.01f);
        return vec3(
            radius * sin(ele) * cos(azimuth),
            radius * cos(ele),
            radius * sin(ele) * sin(azimuth)
        );
    }
} camera;

bool autoRotate = false; // keep off by default

// -------------------- Simple shader helpers --------------------
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint logLen;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 0) {
            std::vector<char> log(logLen);
            glGetShaderInfoLog(s, logLen, nullptr, log.data());
            cerr << "Shader compile error:\n" << log.data() << endl;
        } else {
            cerr << "Shader compile failed (no log)\n";
        }
    }
    return s;
}
static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint logLen; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 0) {
            std::vector<char> log(logLen);
            glGetProgramInfoLog(p, logLen, nullptr, log.data());
            cerr << "Program link error:\n" << log.data() << endl;
        } else {
            cerr << "Program link failed (no log)\n";
        }
    }
    return p;
}

// -------------------- Geometry containers --------------------
struct Pixel {
    float x, y, z;    // world-space or billboard local coords (we'll use model transform)
    float r, g, b, a;
};
struct MeshBuffer {
    vector<Pixel> pixels;
    GLuint vao = 0;
    GLuint vbo = 0;
    int count = 0;
};

// -------------------- Shaders --------------------

// vertex: generic MVP for points; attributes: vec3 pos, vec4 color
const char* vs_points = R"glsl(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
uniform mat4 uMVP;
uniform float uPointSize;
out vec4 vColor;
void main(){
    vColor = aColor;
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
}
)glsl";

// fragment: simple square pixel using color (no circular mask) - used for BH & disk pixels
const char* fs_points = R"glsl(
#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main(){
    // square pixel (no round mask)
    FragColor = vColor;
}
)glsl";

// star shaders: use point sprite gaussian blur using gl_PointCoord
const char* vs_star = R"glsl(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
layout(location=2) in float aSize;
uniform mat4 uMVP;
out vec3 vColor;
out float vSize;
void main(){
    vColor = aColor;
    vSize = aSize;
    gl_Position = uMVP * vec4(aPos,1.0);
    gl_PointSize = aSize;
}
)glsl";

const char* fs_star = R"glsl(
#version 330 core
in vec3 vColor;
in float vSize;
out vec4 FragColor;
void main(){
    // gaussian falloff across point sprite
    vec2 uv = gl_PointCoord - vec2(0.5);
    float r2 = dot(uv,uv);
    // use a slightly wider sigma for stronger blur
    float sigma = 0.32;
    float intensity = exp(-r2 / (2.0 * sigma * sigma));
    intensity = clamp(intensity, 0.0, 1.0);
    vec3 col = vColor * (0.6 + 1.3*intensity);
    FragColor = vec4(col, intensity);
}
)glsl";

// basic grid shader (lines)
const char* vs_grid = R"glsl(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main(){ gl_Position = uMVP * vec4(aPos,1.0); }
)glsl";
const char* fs_grid = R"glsl(
#version 330 core
out vec4 FragColor;
void main(){ FragColor = vec4(0.95,0.75,0.55,1.0); } // warm grid color
)glsl";

// -------------------- Utility: make billboard model --------------------
// Build model that orients billboard so its local XY plane faces the camera.
// We'll use this to render BH pixels (which are defined in local XY plane).
mat4 makeBillboardModel(const vec3 &pos, const vec3 &camPos, float scale) {
    vec3 look = normalize(camPos - pos);
    vec3 right = normalize(cross(vec3(0,1,0), look));
    if (length(right) < 1e-4f) right = normalize(cross(vec3(1,0,0), look));
    vec3 up = cross(look, right);
    mat4 model(1.0f);
    model[0] = vec4(right * scale, 0.0f);
    model[1] = vec4(up    * scale, 0.0f);
    model[2] = vec4(look  * scale, 0.0f);
    model[3] = vec4(pos, 1.0f);
    return model;
}

// -------------------- Generation functions --------------------

// Generate perfectly-packed pixels for black hole circular billboard
void generateBlackHolePixels(MeshBuffer &mb, int res, float radius) {
    mb.pixels.clear();
    // grid sampling mapped to [-radius, radius]
    for (int j = 0; j < res; ++j) {
        for (int i = 0; i < res; ++i) {
            float u = (i + 0.5f) / float(res) * 2.0f - 1.0f;
            float v = (j + 0.5f) / float(res) * 2.0f - 1.0f;
            float x = u * radius;
            float y = v * radius;
            float r = sqrt(x*x + y*y);
            if (r <= radius) {
                Pixel p;
                p.x = x; p.y = y; p.z = 0.0f;
                // color: core black, smooth thin photon transition will be created by photon ring pixels
                // Make core strongly black (as requested)
                p.r = 0.0f; p.g = 0.0f; p.b = 0.0f; p.a = 1.0f;
                mb.pixels.push_back(p);
            }
        }
    }
    mb.count = (int)mb.pixels.size();
}

// Photon ring (thin) - create a thin layered ring in billboard plane
void generatePhotonRingPixels(MeshBuffer &mb, float innerR, float outerR, int samples) {
    mb.pixels.clear();
    // we'll generate two concentric thin radii to give the ring some thickness,
    // but keep it narrow so it reads as "thin ring" distinct from disk.
    float mid = (innerR + outerR) * 0.5f;
    float spread = (outerR - innerR) * 0.45f; // tight spread
    // sample angles densely
    for (int s = 0; s < samples; ++s) {
        float a = (s / float(samples)) * 2.0f * M_PI;
        // sample a couple radii around mid to make ring visible
        for (int layer = 0; layer < 3; ++layer) {
            float layerOffset = ((layer - 1) / 2.0f) * spread * 0.6f; // -0.3..0.3 * spread
            float rr = mid + layerOffset + (((rand()%1000)/1000.0f)-0.5f) * (spread * 0.06f);
            Pixel p;
            p.x = cos(a) * rr;
            p.y = sin(a) * rr;
            p.z = 0.0f;
            // color: more orange than red (user preference)
            float t = (rr - innerR) / (outerR - innerR);
            t = clamp(t, 0.0f, 1.0f);
            vec3 c = mix(vec3(1.0f, 0.48f, 0.08f), vec3(1.0f, 0.18f, 0.02f), t); // orange->red
            // slightly dim toward edges so blending with disk looks smooth
            float alpha = 0.95f * (1.0f - 0.25f * fabs(layer - 1));
            p.r = c.r; p.g = c.g; p.b = c.b; p.a = alpha;
            mb.pixels.push_back(p);
        }
    }
    mb.count = (int)mb.pixels.size();
}

// Disk pixels in horizontal plane (XZ), with small Y variance for thickness.
// We'll create rings with angular samples; output in world coords (not billboard).
// Colors chosen to smoothly blend with photon ring (orange-focused).
void generateDiskPixels(MeshBuffer &mb, float innerR, float outerR, float thickness, int radialSteps, int angularSteps) {
    mb.pixels.clear();
    for (int ri = 0; ri < radialSteps; ++ri) {
        float t = (ri + 0.5f) / float(radialSteps);
        float r = mix(innerR, outerR, t);
        int aSteps = angularSteps;
        for (int ai = 0; ai < aSteps; ++ai) {
            float a = (ai / float(aSteps)) * 2.0f * M_PI;
            // small radial jitter and angular jitter
            float jitterR = (((rand()%1000)/1000.0f) - 0.5f) * 0.0015f; // tiny jitter
            float rr = r + jitterR;
            // Y thickness sampling for small vertical thickness
            // Put most pixels at y=0 (center) and a few at +/- thickness/4 for slight thickness
            for (int layer = 0; layer < 3; ++layer) {
                float yOff = 0.0f;
                if (layer == 0) yOff = 0.0f;
                else if (layer == 1) yOff = thickness * 0.25f;
                else yOff = -thickness * 0.25f;
                Pixel p;
                p.x = cos(a) * rr;
                p.y = yOff;
                p.z = sin(a) * rr;
                // color gradient: inner more red, outer more orange/yellowish
                float mixVal = smoothstep(innerR, outerR, rr);
                // bias to orange tone (user wanted more orange)
                vec3 innerCol = vec3(1.0f, 0.16f, 0.04f); // inner (reddish)
                vec3 outerCol = vec3(1.0f, 0.72f, 0.18f); // outer (orange)
                vec3 col = mix(innerCol, outerCol, pow(mixVal, 0.9f));
                // alpha: stronger near outer to help blend with ring, slightly lower near center
                float alpha = 0.85f * (0.8f + 0.2f * (1.0f - mixVal));
                p.r = col.r; p.g = col.g; p.b = col.b; p.a = alpha;
                mb.pixels.push_back(p);
            }
        }
    }
    mb.count = (int)mb.pixels.size();
}

// -------------------- Setup stars --------------------
void setupStars() {
    stars.clear();
    // left orange-ish, right red-ish; further away and large
    stars.push_back({ vec3( -3.8f, 1.0f, -9.2f ), vec3(1.0f, 0.66f, 0.28f), 88.0f });
    stars.push_back({ vec3(  4.2f, 0.9f, -9.8f ), vec3(1.0f, 0.18f, 0.08f), 92.0f });
}

// -------------------- GL upload helpers --------------------
void uploadPixelMesh(MeshBuffer &mb) {
    if (!mb.vao) glGenVertexArrays(1, &mb.vao);
    if (!mb.vbo) glGenBuffers(1, &mb.vbo);
    glBindVertexArray(mb.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mb.vbo);
    glBufferData(GL_ARRAY_BUFFER, mb.pixels.size() * sizeof(Pixel), mb.pixels.data(), GL_STATIC_DRAW);
    // layout: location 0 = vec3 position, location 1 = vec4 color
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Pixel), (void*)offsetof(Pixel, x));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Pixel), (void*)offsetof(Pixel, r));
    glBindVertexArray(0);
}

// Stars upload (struct separate because different layout)
GLuint starsVAO = 0, starsVBO = 0;
void uploadStars() {
    if (!starsVAO) glGenVertexArrays(1, &starsVAO);
    if (!starsVBO) glGenBuffers(1, &starsVBO);
    glBindVertexArray(starsVAO);
    glBindBuffer(GL_ARRAY_BUFFER, starsVBO);
    glBufferData(GL_ARRAY_BUFFER, stars.size() * sizeof(Star), stars.data(), GL_STATIC_DRAW);
    // layout 0: vec3 pos, layout 1: vec3 color, layout 2: float size
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Star), (void*)offsetof(Star, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Star), (void*)offsetof(Star, color));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Star), (void*)offsetof(Star, size));
    glBindVertexArray(0);
}

// -------------------- Grid (for background) --------------------
struct GridMesh {
    vector<vec3> verts;
    vector<unsigned int> indices;
    GLuint vao = 0, vbo = 0, ebo = 0;
    int indexCount = 0;
};
void generateGrid(GridMesh &g, int gridSize = 28, float spacing = 0.18f, float massScale = 3.0f) {
    g.verts.clear(); g.indices.clear();
    int N = gridSize * 2 + 1;
    for (int z = -gridSize; z <= gridSize; ++z) {
        for (int x = -gridSize; x <= gridSize; ++x) {
            float fx = x * spacing;
            float fz = z * spacing;
            float r = sqrt(fx*fx + fz*fz);
            // visual Schwarzschild-like dip
            float A = 0.45f * massScale;
            float fy = -A / (r + 0.08f) * exp(-r*0.6f);
            g.verts.emplace_back(fx, fy - 0.28f, fz); // shift so BH sits nicely
        }
    }
    for (int z = 0; z < N; ++z) {
        for (int x = 0; x < N; ++x) {
            int i = z*N + x;
            if (x < N-1) { g.indices.push_back(i); g.indices.push_back(i+1); }
            if (z < N-1) { g.indices.push_back(i); g.indices.push_back(i+N); }
        }
    }
    g.indexCount = (int)g.indices.size();
    if (!g.vao) glGenVertexArrays(1,&g.vao);
    if (!g.vbo) glGenBuffers(1,&g.vbo);
    if (!g.ebo) glGenBuffers(1,&g.ebo);
    glBindVertexArray(g.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g.vbo);
    glBufferData(GL_ARRAY_BUFFER, g.verts.size()*sizeof(vec3), g.verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, g.indices.size()*sizeof(unsigned int), g.indices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}

// -------------------- Callbacks --------------------
void mouse_button_cb(GLFWwindow* w, int button, int action, int mods){
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        camera.dragging = (action == GLFW_PRESS);
        if (camera.dragging) glfwGetCursorPos(w, &camera.lastX, &camera.lastY);
    }
}
void cursor_pos_cb(GLFWwindow* w, double x, double y){
    if (camera.dragging) {
        float dx = float(x - camera.lastX);
        float dy = float(y - camera.lastY);
        camera.azimuth += dx * camera.orbitSpeed;
        camera.elevation -= dy * camera.orbitSpeed;
        camera.lastX = x; camera.lastY = y;
    }
}
void scroll_cb(GLFWwindow* w, double xoff, double yoff){
    camera.radius -= float(yoff) * camera.zoomSpeed;
    camera.radius = clamp(camera.radius, 0.8f, 50.0f);
}
void key_cb(GLFWwindow* w, int key, int sc, int action, int mods){
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(w, true);
    if (key == GLFW_KEY_R && action == GLFW_PRESS) autoRotate = !autoRotate;
    if (key == GLFW_KEY_UP && (action==GLFW_PRESS||action==GLFW_REPEAT)) {
        BH_PIXEL_RES = glm::min(1024, BH_PIXEL_RES + 16);
        cout << "BH_PIXEL_RES = " << BH_PIXEL_RES << endl;
    }
    if (key == GLFW_KEY_DOWN && (action==GLFW_PRESS||action==GLFW_REPEAT)) {
        BH_PIXEL_RES = glm::max(16, BH_PIXEL_RES - 16);
        cout << "BH_PIXEL_RES = " << BH_PIXEL_RES << endl;
    }
}

// -------------------- main --------------------
int main() {
    srand((unsigned)time(nullptr));
    if (!glfwInit()) { cerr << "GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(WIN_W, WIN_H, "Pixel Black Hole - fused ring + disk (orange)", nullptr, nullptr);
    if (!win) { cerr << "Window creation failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { cerr << "GLEW init failed\n"; glfwTerminate(); return -1; }

    // callbacks
    glfwSetMouseButtonCallback(win, mouse_button_cb);
    glfwSetCursorPosCallback(win, cursor_pos_cb);
    glfwSetScrollCallback(win, scroll_cb);
    glfwSetKeyCallback(win, key_cb);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // compile programs
    GLuint vs_p = compileShader(GL_VERTEX_SHADER, vs_points);
    GLuint fs_p = compileShader(GL_FRAGMENT_SHADER, fs_points);
    GLuint progPoints = linkProgram(vs_p, fs_p);

    GLuint vs_s = compileShader(GL_VERTEX_SHADER, vs_star);
    GLuint fs_s = compileShader(GL_FRAGMENT_SHADER, fs_star);
    GLuint progStar = linkProgram(vs_s, fs_s);

    GLuint vs_g = compileShader(GL_VERTEX_SHADER, vs_grid);
    GLuint fs_g = compileShader(GL_FRAGMENT_SHADER, fs_grid);
    GLuint progGrid = linkProgram(vs_g, fs_g);

    // generate scene geometry
    GridMesh grid; generateGrid(grid, 28, 0.12f, 3.2f);

    MeshBuffer bhPixels; generateBlackHolePixels(bhPixels, BH_PIXEL_RES, BH_RADIUS);
    MeshBuffer photonRing; generatePhotonRingPixels(photonRing, PH_RING_IN, PH_RING_OUT, PHOTON_SAMPLES);
    MeshBuffer diskPixels; generateDiskPixels(diskPixels, DISK_INNER, DISK_OUTER, DISK_THICKNESS, DISK_RADIAL_STEPS, DISK_ANGULAR_STEPS);

    // prepare stars
    setupStars();
    uploadStars();

    // upload pixel buffers
    uploadPixelMesh(bhPixels);
    uploadPixelMesh(photonRing);
    uploadPixelMesh(diskPixels);

    // projection
    mat4 proj = perspective(radians(60.0f), float(WIN_W)/float(WIN_H), 0.1f, 200.0f);

    // uniforms locations
    GLint loc_uMVP_points = glGetUniformLocation(progPoints, "uMVP");
    GLint loc_pointSize = glGetUniformLocation(progPoints, "uPointSize");

    GLint loc_uMVP_star = glGetUniformLocation(progStar, "uMVP");

    GLint loc_uMVP_grid = glGetUniformLocation(progGrid, "uMVP");

    // main loop
    double lastT = glfwGetTime();
    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        double dt = now - lastT;
        lastT = now;

        if (!camera.dragging && autoRotate) camera.azimuth += 0.0009f;

        // clear
        glViewport(0,0,WIN_W,WIN_H);
        glClearColor(0.02f, 0.01f, 0.01f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // view/proj
        vec3 camPos = camera.position();
        mat4 view = lookAt(camPos, camera.target, vec3(0,1,0));
        mat4 VP = proj * view;

        // draw grid
        glUseProgram(progGrid);
        if (loc_uMVP_grid >= 0) glUniformMatrix4fv(loc_uMVP_grid, 1, GL_FALSE, value_ptr(VP));
        glBindVertexArray(grid.vao);
        glDrawElements(GL_LINES, grid.indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // draw disk pixels (horizontal) -> place in world with slight downward offset so disk sits around BH
        glUseProgram(progPoints);
        // diskModel: translate to black hole center y shift (we want disk horizontal in world space)
        mat4 diskModel = translate(mat4(1.0f), vec3(0.0f, -0.03f, 0.0f));
        mat4 diskMVP = VP * diskModel;
        if (loc_uMVP_points >= 0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(diskMVP));
        if (loc_pointSize >= 0) glUniform1f(loc_pointSize, pixelPointSize * 6.0f); // disk pixels smaller to avoid visible gaps
        glBindVertexArray(diskPixels.vao);
        glDrawArrays(GL_POINTS, 0, diskPixels.count);
        glBindVertexArray(0);

        // draw photon ring (billboard relative to camera) - thin ring just outside BH
        mat4 ringModel = makeBillboardModel(vec3(0.0f, -0.02f, 0.0f), camPos, 1.0f);
        mat4 ringMVP = VP * ringModel;
        if (loc_uMVP_points >= 0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(ringMVP));
        if (loc_pointSize >= 0) glUniform1f(loc_pointSize, pixelPointSize * 0.9f);
        glBindVertexArray(photonRing.vao);
        glDrawArrays(GL_POINTS, 0, photonRing.count);
        glBindVertexArray(0);

        // draw black hole center pixels ON TOP (billboard) - black pixels will occlude disk visually
        // BH should be large and mostly black; ensure BH pixels drawn after disk to occlude properly
        mat4 bhModel = makeBillboardModel(vec3(0.0f, 0.0f, 0.0f), camPos, 1.0f);
        mat4 bhMVP = VP * bhModel;
        if (loc_uMVP_points >= 0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(bhMVP));
        if (loc_pointSize >= 0) glUniform1f(loc_pointSize, pixelPointSize);
        glBindVertexArray(bhPixels.vao);
        glDrawArrays(GL_POINTS, 0, bhPixels.count);
        glBindVertexArray(0);

        // draw stars (point sprites with gaussian blur)
        glUseProgram(progStar);
        if (loc_uMVP_star >= 0) glUniformMatrix4fv(loc_uMVP_star, 1, GL_FALSE, value_ptr(VP));
        glBindVertexArray(starsVAO);
        // each star is a single vertex in the buffer
        glDrawArrays(GL_POINTS, 0, (GLsizei)stars.size());
        glBindVertexArray(0);

        // swap/poll
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    // cleanup
    if (grid.vao) glDeleteVertexArrays(1, &grid.vao);
    if (grid.vbo) glDeleteBuffers(1, &grid.vbo);
    if (grid.ebo) glDeleteBuffers(1, &grid.ebo);

    if (bhPixels.vao) glDeleteVertexArrays(1, &bhPixels.vao);
    if (bhPixels.vbo) glDeleteBuffers(1, &bhPixels.vbo);
    if (photonRing.vao) glDeleteVertexArrays(1, &photonRing.vao);
    if (photonRing.vbo) glDeleteBuffers(1, &photonRing.vbo);
    if (diskPixels.vao) glDeleteVertexArrays(1, &diskPixels.vao);
    if (diskPixels.vbo) glDeleteBuffers(1, &diskPixels.vbo);

    if (starsVAO) glDeleteVertexArrays(1, &starsVAO);
    if (starsVBO) glDeleteBuffers(1, &starsVBO);

    glDeleteProgram(progPoints);
    glDeleteProgram(progStar);
    glDeleteProgram(progGrid);

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

