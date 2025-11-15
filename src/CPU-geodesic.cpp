// main.cpp
// Pixelated BH + horizontal pixel accretion disk + visible photon ring (billboard)
// + blurred stars whose light rays are curved by a GPU post-process.
// OpenGL 3.3 core, needs GLFW, GLEW, glm.
// Compile example:
// g++ main.cpp -lglfw -lGLEW -lGL -ldl -pthread -o blackhole
//
// Large single-file implementation with many inline comments so you can tune easily.

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

// ========================================================
// ================ Scene / Tweak parameters ==============
// ========================================================

const int WIN_W = 1280;
const int WIN_H = 800;

// Pixel resolution for the black hole (higher -> denser pixels)
int BH_PIXEL_RES = 128;        // can increase (256, 384...) if needed
float BH_RADIUS = 0.65f;       // billboard radius units (scene units)
float pixelPointSize = 6.0f;   // size in screen pixels for each "block"

// Disk parameters (horizontal plane, centered at blackPos.y)
int DISK_RADIAL_STEPS = 36;
int DISK_ANGULAR_STEPS = 360;
float DISK_INNER = 0.50f;
float DISK_OUTER = 0.95f;
float DISK_THICKNESS = 0.04f;

// Photon ring billboard radii (in billboard local units)
float PH_RING_IN = BH_RADIUS * 0.8f;
float PH_RING_OUT = BH_RADIUS * 0.95f;
int PH_RING_SAMPLES = 720;

// Stars
struct Star { vec3 pos; vec3 color; float size; };
vector<Star> stars;

// Camera
struct Camera {
    float radius = 3.5f;
    float azimuth = 0.0f;
    float elevation = glm::pi<float>()/2.0f;
    bool dragging = false;
    double lastX=0, lastY=0;
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

bool autoRotate = false;

// ============== Star-warp post-process parameters (GPU) ==============
// Tweak to change how much the star rays curve
float STAR_WARP_STRENGTH = 0.15f;   // overall strength (increase -> more bending)
float STAR_WARP_FALLOFF = 2.0f;     // falloff exponent
float BH_SCREEN_EFFECT_RADIUS = 0.22f; // normalized screen radius (0..1) around BH where effect is strong
float STAR_RING_SHARPNESS = 8.0f;   // controls how ring-like the warped light becomes

// ========================================================
// ================= Shader helpers =======================
// ========================================================
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok) {
        GLint logLen; glGetShaderiv(s, GL_INFO_LOG_LENGTH, &logLen);
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
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok) {
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

// ========================================================
// ================= Geometry / mesh utils =================
// ========================================================

struct Pixel {
    float x,y,z;
    float r,g,b,a;
};
struct MeshBuffer {
    vector<Pixel> pixels;
    GLuint vao=0, vbo=0;
    int count=0;
};

struct GridMesh { vector<vec3> verts; vector<unsigned int> indices; GLuint vao=0,vbo=0,ebo=0; int indexCount=0; };

// Billboard helper (same as you used before)
mat4 makeBillboardModel(const vec3 &pos, const vec3 &camPos, float scale){
    vec3 look = normalize(camPos - pos);
    vec3 right = normalize(cross(vec3(0,1,0), look));
    if(length(right) < 1e-4f) right = normalize(cross(vec3(1,0,0), look));
    vec3 up = cross(look, right);
    mat4 model(1.0f);
    model[0] = vec4(right*scale, 0.0f);
    model[1] = vec4(up*scale, 0.0f);
    model[2] = vec4(look*scale, 0.0f);
    model[3] = vec4(pos, 1.0f);
    return model;
}

// ========================================================
// ================= Mesh generation =======================
// ========================================================

// Black hole interior pixels (billboard local)
void generateBlackHolePixels(MeshBuffer &mb, int res, float radius) {
    mb.pixels.clear();
    for (int j=0;j<res;++j){
        for (int i=0;i<res;++i){
            float u = (i + 0.5f)/float(res)*2.0f - 1.0f;
            float v = (j + 0.5f)/float(res)*2.0f - 1.0f;
            float x = u * radius;
            float y = v * radius;
            float r = sqrt(x*x + y*y);
            if (r <= radius) {
                Pixel p;
                p.x = x; p.y = y; p.z = 0.01f;
                p.r = 0.0f; p.g = 0.0f; p.b = 0.0f; p.a = 1.0f;
                mb.pixels.push_back(p);
            }
        }
    }
    mb.count = (int)mb.pixels.size();
}

// Photon ring (billboard-local)
void generatePhotonRingBillboard(MeshBuffer &mb, float inR, float outR, int samples) {
    mb.pixels.clear();
    float radialStep = (outR - inR) / 6.0f;
    for (int s=0;s<samples;++s){
        float a = (s + 0.5f)/float(samples) * 2.0f * M_PI;
        for (float r = inR; r <= outR; r += radialStep){
            float jitter = ((rand()%1000)/1000.0f - 0.5f) * 0.004f;
            float rr = r + jitter;
            Pixel p;
            p.x = cos(a) * rr;
            p.y = sin(a) * rr;
            p.z = 0.0f;
            float tt = (rr - inR) / (outR - inR);
            vec3 col = mix(vec3(1.0f, 0.55f, 0.08f), vec3(1.0f, 0.12f, 0.02f), tt);
            float alpha = 0.95f * (0.6f + 0.6f * (1.0f - abs(tt - 0.5f)));
            p.r = col.r; p.g = col.g; p.b = col.b; p.a = alpha;
            mb.pixels.push_back(p);
        }
    }
    mb.count = (int)mb.pixels.size();
}

// Disk pixels in world coordinates (horizontal)
void generateDiskPixelsWorld(MeshBuffer &mb, float innerR, float outerR, float thickness, int radialSteps, int angularSteps, float blackY) {
    mb.pixels.clear();
    for (int ri=0; ri<radialSteps; ++ri){
        float t = (ri+0.5f)/float(radialSteps);
        float r = mix(innerR, outerR, t);
        int aSteps = angularSteps;
        for (int ai=0; ai<aSteps; ++ai){
            float a = (ai + 0.5f)/float(aSteps) * 2.0f * M_PI;
            float jitter = ((rand()%1000)/1000.0f - 0.5f) * 0.003f;
            float rr = r + jitter;
            for (int yi=0; yi<3; ++yi) {
                float y = blackY + (-thickness*0.35f + yi * (thickness*0.35f));
                Pixel p;
                p.x = cos(a)*rr;
                p.z = sin(a)*rr;
                p.y = y;
                float radialNorm = smoothstep(innerR, outerR, rr);
                vec3 innerColor = vec3(0.96f, 0.12f, 0.03f);
                vec3 outerColor = vec3(1.0f, 0.78f, 0.18f);
                vec3 col = mix(innerColor, outerColor, radialNorm);
                float alpha = 0.92f * (0.6f + 0.6f*radialNorm);
                p.r = col.r; p.g = col.g; p.b = col.b; p.a = alpha;
                mb.pixels.push_back(p);
            }
        }
    }
    mb.count = (int)mb.pixels.size();
}

// Grid for background (Schwarzschild-like)
void generateGrid(GridMesh &g, int gridSize=28, float spacing=0.12f, float massScale=3.2f) {
    g.verts.clear(); g.indices.clear();
    int N = gridSize*2 + 1;
    for (int z=-gridSize; z<=gridSize; ++z){
        for (int x=-gridSize; x<=gridSize; ++x){
            float fx = x * spacing;
            float fz = z * spacing;
            float r = sqrt(fx*fx + fz*fz);
            float A = 0.45f * massScale;
            float fy = -A / (r + 0.08f) * exp(-r*0.6f);
            g.verts.emplace_back(fx, fy - 0.28f, fz);
        }
    }
    for (int z=0; z<N; ++z){
        for (int x=0; x<N; ++x){
            int i = z*N + x;
            if (x < N-1) { g.indices.push_back(i); g.indices.push_back(i+1); }
            if (z < N-1) { g.indices.push_back(i); g.indices.push_back(i+N); }
        }
    }
    g.indexCount = (int)g.indices.size();
    if (!g.vao) glGenVertexArrays(1, &g.vao);
    if (!g.vbo) glGenBuffers(1, &g.vbo);
    if (!g.ebo) glGenBuffers(1, &g.ebo);
    glBindVertexArray(g.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g.vbo);
    glBufferData(GL_ARRAY_BUFFER, g.verts.size()*sizeof(vec3), g.verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, g.indices.size()*sizeof(unsigned int), g.indices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}

// Stars setup
void setupStars() {
    stars.clear();
    // Put them a little closer (user request) and colored orange/red
    stars.push_back({ vec3(-3.8f, 0.9f, -7.8f), vec3(1.0f, 0.66f, 0.32f), 72.0f });
    stars.push_back({ vec3( 4.2f, 0.7f, -8.3f), vec3(1.0f, 0.18f, 0.08f), 84.0f });
}

// ========================================================
// ================= Upload helpers =======================
// ========================================================

void uploadMesh(MeshBuffer &mb) {
    if (!mb.vao) glGenVertexArrays(1, &mb.vao);
    if (!mb.vbo) glGenBuffers(1, &mb.vbo);
    glBindVertexArray(mb.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mb.vbo);
    glBufferData(GL_ARRAY_BUFFER, mb.pixels.size()*sizeof(Pixel), mb.pixels.data(), GL_STATIC_DRAW);
    // layout 0: vec3 pos, 1: vec4 color
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Pixel),(void*)offsetof(Pixel,x));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,sizeof(Pixel),(void*)offsetof(Pixel,r));
    glBindVertexArray(0);
}

// Stars upload (separate layout)
GLuint starsVAO = 0, starsVBO = 0;
void uploadStars() {
    if(!starsVAO) glGenVertexArrays(1, &starsVAO);
    if(!starsVBO) glGenBuffers(1, &starsVBO);
    glBindVertexArray(starsVAO);
    glBindBuffer(GL_ARRAY_BUFFER, starsVBO);
    glBufferData(GL_ARRAY_BUFFER, stars.size()*sizeof(Star), stars.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Star),(void*)offsetof(Star,pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Star),(void*)offsetof(Star,color));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2,1,GL_FLOAT,GL_FALSE,sizeof(Star),(void*)offsetof(Star,size));
    glBindVertexArray(0);
}

// ========================================================
// ================= Shaders sources =======================
// ========================================================

// Basic vertex/fragment for grid / simple points
const char* vs_basic = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main(){ gl_Position = uMVP * vec4(aPos,1.0); }
)GLSL";

const char* fs_grid = R"GLSL(
#version 330 core
out vec4 FragColor;
void main(){ FragColor = vec4(0.95,0.7,0.45,1.0); }
)GLSL";

// Points (pixels) shader: input vec3 pos, vec4 color
const char* vs_points = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;
uniform mat4 uMVP;
uniform float uPointSize;
out vec4 vCol;
void main(){
    vCol = aCol;
    gl_Position = uMVP * vec4(aPos,1.0);
    gl_PointSize = uPointSize;
}
)GLSL";

const char* fs_points = R"GLSL(
#version 330 core
in vec4 vCol;
out vec4 FragColor;
void main(){
    // square pixel - no circular mask to keep pixelated look
    FragColor = vCol;
}
)GLSL";

// Stars vertex/fragment (point sprites, gaussian blur)
const char* vs_star = R"GLSL(
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
)GLSL";

const char* fs_star = R"GLSL(
#version 330 core
in vec3 vColor;
in float vSize;
out vec4 FragColor;
void main(){
    // round gaussian star (avoid square blur)
    vec2 uv = gl_PointCoord - vec2(0.5);
    float r2 = dot(uv,uv);
    float sigma = 0.18; // smaller -> tighter round star
    float intensity = exp(-r2 / (2.0 * sigma * sigma));
    intensity = clamp(intensity, 0.0, 1.0);
    vec3 col = vColor * (0.5 + 1.1 * intensity);
    FragColor = vec4(col, intensity);
}
)GLSL";

// Fullscreen quad vertex for postprocess
const char* vs_quad = R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main(){
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

// Postprocess fragment: warp star texture by BH-screen position
// Approach (approximate, artistic):
// - compute vector from current fragment uv to BH screen uv
// - compute distance (impact parameter) d
// - compute deflection magnitude = strength * (ringRadius / (d + eps))^power * falloff
// - deflect sample towards a tangent direction (perp) to create arcs (not only radial shift)
// - combine radial and tangential components and sample starTex with offset
// This creates ring-like curved rays depending on BH position and strength.
const char* fs_warp_stars = R"GLSL(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uStarsTex; // rendered stars
uniform vec2 uBH_UV;         // BH position in screen-space UV (0..1)
uniform float uStrength;     // general strength
uniform float uFalloff;      // falloff exponent
uniform float uEffectRadius; // radius (0..1) of influence
uniform float uRingSharpness;// how ringy the warp becomes
uniform vec3 uStarColorsMask; // optional mask, not used here
void main(){
    vec2 uv = vUV;
    vec2 toBH = uv - uBH_UV;
    float d = length(toBH);
    float eps = 1e-4;

    // If outside a region, sample original (no warp) - keeps grid/disk unaffected (we render only star texture here)
    if (d > uEffectRadius*1.8) {
        FragColor = texture(uStarsTex, uv);
        return;
    }

    // normalized direction from BH to frag
    vec2 dir = normalize(toBH + vec2(eps));

    // tangential (perpendicular) direction -> create arc when moving sample along tangent
    vec2 tang = vec2(-dir.y, dir.x);

    // compute a ring radius in screen-space (small offset)
    float ringR = uEffectRadius * 0.9;

    // base deflection magnitude (radial-inward scaling)
    // stronger near BH center, weaker far away
    float base = uStrength * pow(max(0.001, (uEffectRadius / (d + 0.001))), uFalloff);

    // create a peaked ring contribution so light concentrates into an arc near ringR
    // gaussian-like peak centered on ringR to let arcs form around ring
    float ringPeak = exp( - ( (d - ringR)*(d - ringR) ) * uRingSharpness );

    // radial component (moves sample radially towards BH)
    vec2 radialShift = -dir * base * 0.25 * ringPeak;

    // tangential component (moves sample perpendicular to create arc)
    // choose sign based on screen-space y to create asymmetry (or can be zero-mean jitter)
    float sign = 1.0;
    if (uBH_UV.y < 0.5) sign = -1.0; // small heuristic for nicer arcs
    vec2 tangentialShift = tang * base * 0.6 * ringPeak * sign;

    // combined offset (in UV space) - scale down to avoid extreme warps
    vec2 offset = (radialShift + tangentialShift) * 0.5;

    // final sample
    vec4 sample = texture(uStarsTex, uv + offset);

    // increase intensity near ring (bloom-like) - artistic
    float boost = 1.0 + 0.8 * ringPeak * smoothstep(uEffectRadius*0.02, uEffectRadius*0.9, d);

    // optionally fade depending on d (so very close fragments are dominated by BH occlusion)
    sample.rgb *= boost;

    FragColor = sample;
}
)GLSL";

// ========================================================
// ================= Full program =========================
// ========================================================

// input quad for postprocess
static const float quadVerts[] = {
    -1.0f, -1.0f,
     1.0f, -1.0f,
    -1.0f,  1.0f,
     1.0f,  1.0f
};

// ========================================================
// ===================== Callbacks =========================
// ========================================================
void mouse_button_cb(GLFWwindow* w, int button, int action, int mods){
    if(button == GLFW_MOUSE_BUTTON_LEFT){
        camera.dragging = (action == GLFW_PRESS);
        if(camera.dragging) glfwGetCursorPos(w, &camera.lastX, &camera.lastY);
    }
}
void cursor_pos_cb(GLFWwindow* w, double x, double y){
    if(camera.dragging){
        float dx = float(x - camera.lastX);
        float dy = float(y - camera.lastY);
        camera.azimuth += dx * camera.orbitSpeed;
        camera.elevation -= dy * camera.orbitSpeed;
        camera.lastX = x; camera.lastY = y;
    }
}
void scroll_cb(GLFWwindow* w, double xoff, double yoff){
    camera.radius -= float(yoff) * camera.zoomSpeed;
    camera.radius = clamp(camera.radius, 0.5f, 100.0f);
}
void key_cb(GLFWwindow* w, int key, int sc, int action, int mods){
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(w, true);
    if (key == GLFW_KEY_R && action == GLFW_PRESS) autoRotate = !autoRotate;
    if (key == GLFW_KEY_UP && (action==GLFW_PRESS||action==GLFW_REPEAT)) {
        BH_PIXEL_RES = glm::min(1024, BH_PIXEL_RES + 16);
        cerr << "BH_PIXEL_RES = " << BH_PIXEL_RES << endl;
    }
    if (key == GLFW_KEY_DOWN && (action==GLFW_PRESS||action==GLFW_REPEAT)) {
        BH_PIXEL_RES = glm::max(16, BH_PIXEL_RES - 16);
        cerr << "BH_PIXEL_RES = " << BH_PIXEL_RES << endl;
    }
    if (key == GLFW_KEY_KP_ADD && (action==GLFW_PRESS||action==GLFW_REPEAT)) {
        pixelPointSize = glm::min(64.0f, pixelPointSize + 1.0f);
        cerr << "pixelPointSize = " << pixelPointSize << endl;
    }
    if (key == GLFW_KEY_KP_SUBTRACT && (action==GLFW_PRESS||action==GLFW_REPEAT)) {
        pixelPointSize = glm::max(1.0f, pixelPointSize - 1.0f);
        cerr << "pixelPointSize = " << pixelPointSize << endl;
    }
}

// ========================================================
// ====================== Main =============================
// ========================================================
int main() {
    srand((unsigned)time(nullptr));
    if (!glfwInit()) { cerr<<"GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(WIN_W, WIN_H, "Pixel Black Hole - GPU star ray warp", nullptr, nullptr);
    if (!win) { cerr<<"Window creation failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { cerr<<"GLEW init failed\n"; glfwTerminate(); return -1; }

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
    GLuint vs_g = compileShader(GL_VERTEX_SHADER, vs_basic);
    GLuint fs_g = compileShader(GL_FRAGMENT_SHADER, fs_grid);
    GLuint progGrid = linkProgram(vs_g, fs_g);

    GLuint vsP = compileShader(GL_VERTEX_SHADER, vs_points);
    GLuint fsP = compileShader(GL_FRAGMENT_SHADER, fs_points);
    GLuint progPoints = linkProgram(vsP, fsP);

    GLuint vsS = compileShader(GL_VERTEX_SHADER, vs_star);
    GLuint fsS = compileShader(GL_FRAGMENT_SHADER, fs_star);
    GLuint progStar = linkProgram(vsS, fsS);

    GLuint vsQ = compileShader(GL_VERTEX_SHADER, vs_quad);
    GLuint fsWarp = compileShader(GL_FRAGMENT_SHADER, fs_warp_stars);
    GLuint progWarp = linkProgram(vsQ, fsWarp);

    // set up full-screen quad VAO for postprocess
    GLuint quadVAO=0, quadVBO=0;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(float),(void*)0);
    glBindVertexArray(0);

    // generate scene geometry
    GridMesh grid; generateGrid(grid, 28, 0.12f, 3.2f);

    MeshBuffer bhPixels, diskPixels, ringPixels;
    generateBlackHolePixels(bhPixels, BH_PIXEL_RES, BH_RADIUS);

    vec3 blackPos = vec3(0.0f, -0.28f, 0.0f);
    generateDiskPixelsWorld(diskPixels, DISK_INNER, DISK_OUTER, DISK_THICKNESS,
                            DISK_RADIAL_STEPS, DISK_ANGULAR_STEPS, blackPos.y);

    generatePhotonRingBillboard(ringPixels, PH_RING_IN, PH_RING_OUT, PH_RING_SAMPLES);

    setupStars();
    uploadStars();

    // upload pixel buffers
    uploadMesh(bhPixels);
    uploadMesh(diskPixels);
    uploadMesh(ringPixels);

    // prepare star FBO: render only stars to texture, then apply warp postprocess
    GLuint starsFBO=0, starsTex=0, starsRBO=0;
    glGenFramebuffers(1, &starsFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, starsFBO);

    glGenTextures(1, &starsTex);
    glBindTexture(GL_TEXTURE_2D, starsTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIN_W, WIN_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // smoothing is ok for stars layer
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, starsTex, 0);

    // depth renderbuffer (we don't need depth for stars layer but keep it to ensure correct behavior)
    glGenRenderbuffers(1, &starsRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, starsRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIN_W, WIN_H);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, starsRBO);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        cerr << "Star FBO incomplete\n";
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // projection
    mat4 proj = perspective(radians(60.0f), float(WIN_W)/float(WIN_H), 0.1f, 300.0f);

    // uniform locations
    GLint loc_uMVP_points = glGetUniformLocation(progPoints, "uMVP");
    GLint loc_pointSize = glGetUniformLocation(progPoints, "uPointSize");
    GLint loc_uMVP_star = glGetUniformLocation(progStar, "uMVP");
    GLint loc_uMVP_grid = glGetUniformLocation(progGrid, "uMVP");

    GLint loc_warp_bhUV = glGetUniformLocation(progWarp, "uBH_UV");
    GLint loc_warp_strength = glGetUniformLocation(progWarp, "uStrength");
    GLint loc_warp_falloff = glGetUniformLocation(progWarp, "uFalloff");
    GLint loc_warp_radius = glGetUniformLocation(progWarp, "uEffectRadius");
    GLint loc_warp_ringsharp = glGetUniformLocation(progWarp, "uRingSharpness");
    GLint loc_warp_starsTex = glGetUniformLocation(progWarp, "uStarsTex");

    // star program MVP location
    GLint loc_star_uMVP = glGetUniformLocation(progStar, "uMVP");

    // main loop
    while (!glfwWindowShouldClose(win)) {
        // basic updates
        if (!camera.dragging && autoRotate) camera.azimuth += 0.0009f;

        // view/proj
        vec3 camPos = camera.position();
        mat4 view = lookAt(camPos, camera.target, vec3(0,1,0));
        mat4 VP = proj * view;

        // ---------------------------
        // 1) Render stars into FBO (only stars)
        // ---------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, starsFBO);
        glViewport(0,0,WIN_W,WIN_H);
        glClearColor(0.0f,0.0f,0.0f,0.0f); // transparent background so we can composite
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // draw stars as point sprites
        glUseProgram(progStar);
        if (loc_star_uMVP >= 0) glUniformMatrix4fv(loc_star_uMVP, 1, GL_FALSE, value_ptr(VP));
        glBindVertexArray(starsVAO);
        glDrawArrays(GL_POINTS, 0, (GLsizei)stars.size());
        glBindVertexArray(0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0); // back to default

        // ---------------------------
        // 2) Clear default framebuffer and draw grid/disk/BH (background + foreground)
        // ---------------------------
        glViewport(0,0,WIN_W,WIN_H);
        glClearColor(0.02f, 0.01f, 0.01f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // draw grid (lines)
        glUseProgram(progGrid);
        if (loc_uMVP_grid >= 0) glUniformMatrix4fv(loc_uMVP_grid, 1, GL_FALSE, value_ptr(VP));
        glBindVertexArray(grid.vao);
        glDrawElements(GL_LINES, grid.indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // draw disk (world horizontal)
        glUseProgram(progPoints);
        mat4 diskModel = translate(mat4(1.0f), vec3(0.0f, 0.0f, 0.0f)); // disk pixels already at blackPos.y
        mat4 diskMVP = VP * diskModel;
        if (loc_uMVP_points >= 0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(diskMVP));
        if (loc_pointSize >= 0) glUniform1f(loc_pointSize, pixelPointSize * 1.25f);
        glBindVertexArray(diskPixels.vao);
        glDrawArrays(GL_POINTS, 0, diskPixels.count);
        glBindVertexArray(0);

        // draw photon ring billboard (slightly outside) - will be composited on top of the (warped) star layer later visually
        mat4 ringModel = makeBillboardModel(vec3(blackPos.x, blackPos.y + 0.0f, blackPos.z), camPos, 1.0f);
        mat4 ringMVP = VP * ringModel;
        if (loc_uMVP_points >= 0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(ringMVP));
        if (loc_pointSize >= 0) glUniform1f(loc_pointSize, pixelPointSize * 0.95f);
        glBindVertexArray(ringPixels.vao);
        glDrawArrays(GL_POINTS, 0, ringPixels.count);
        glBindVertexArray(0);

        // draw BH billboard center on top so it occludes disk center
        mat4 bhModel = makeBillboardModel(blackPos, camPos, 0.7f);
        mat4 bhMVP = VP * bhModel;
        if (loc_uMVP_points >= 0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(bhMVP));
        if (loc_pointSize >= 0) glUniform1f(loc_pointSize, pixelPointSize);
        glBindVertexArray(bhPixels.vao);
        glDrawArrays(GL_POINTS, 0, bhPixels.count);
        glBindVertexArray(0);

        // ---------------------------
        // 3) Render warped stars (postprocess) onto screen **behind** BH center pixels visually
        //    We already drew BH and ring above; but we want stars to appear behind BH center,
        //    so we render warped stars now and then redraw BH center on top to ensure occlusion.
        //    To keep layering consistent: we will blend the warped stars over background (grid/disk),
        //    then draw BH center again (occlusion), then possibly some subtle compositing.
        // ---------------------------

        // bind star texture to unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, starsTex);

        // prepare shader
        glUseProgram(progWarp);
        glUniform1i(loc_warp_starsTex, 0);
        // compute BH screen-space UV: project blackPos into clip space then to 0..1
        vec4 bhClip = VP * vec4(blackPos, 1.0f);
        vec3 bhNDC = vec3(bhClip) / bhClip.w;
        vec2 bhUV = vec2(bhNDC.x * 0.5f + 0.5f, bhNDC.y * 0.5f + 0.5f);
        if (loc_warp_bhUV >= 0) glUniform2fv(loc_warp_bhUV, 1, value_ptr(bhUV));
        if (loc_warp_strength >= 0) glUniform1f(loc_warp_strength, STAR_WARP_STRENGTH);
        if (loc_warp_falloff >= 0) glUniform1f(loc_warp_falloff, STAR_WARP_FALLOFF);
        if (loc_warp_radius >= 0) glUniform1f(loc_warp_radius, BH_SCREEN_EFFECT_RADIUS);
        if (loc_warp_ringsharp >= 0) glUniform1f(loc_warp_ringsharp, STAR_RING_SHARPNESS);

        // draw full-screen quad
        glBindVertexArray(quadVAO);
        // enable blending so warped stars composite nicely
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);

        // re-draw BH center on top to ensure it fully occludes star rays behind it
        glUseProgram(progPoints);
        if (loc_uMVP_points >= 0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(bhMVP));
        if (loc_pointSize >= 0) glUniform1f(loc_pointSize, pixelPointSize);
        glBindVertexArray(bhPixels.vao);
        glDrawArrays(GL_POINTS, 0, bhPixels.count);
        glBindVertexArray(0);

        // Optionally: draw ring again with additive blending for glow (small)
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        if (loc_uMVP_points >= 0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(ringMVP));
        if (loc_pointSize >= 0) glUniform1f(loc_pointSize, pixelPointSize * 1.05f);
        glBindVertexArray(ringPixels.vao);
        glDrawArrays(GL_POINTS, 0, ringPixels.count);
        glBindVertexArray(0);

        // draw stars (non-warped) faintly on top for artistic effect? skip to avoid double-counting.

        // swap
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    // cleanup
    if (grid.vao) glDeleteVertexArrays(1, &grid.vao);
    if (grid.vbo) glDeleteBuffers(1, &grid.vbo);
    if (grid.ebo) glDeleteBuffers(1, &grid.ebo);

    if (bhPixels.vao) glDeleteVertexArrays(1, &bhPixels.vao);
    if (bhPixels.vbo) glDeleteBuffers(1, &bhPixels.vbo);
    if (diskPixels.vao) glDeleteVertexArrays(1, &diskPixels.vao);
    if (diskPixels.vbo) glDeleteBuffers(1, &diskPixels.vbo);
    if (ringPixels.vao) glDeleteVertexArrays(1, &ringPixels.vao);
    if (ringPixels.vbo) glDeleteBuffers(1, &ringPixels.vbo);

    if (starsVAO) glDeleteVertexArrays(1, &starsVAO);
    if (starsVBO) glDeleteBuffers(1, &starsVBO);

    if (quadVAO) glDeleteVertexArrays(1, &quadVAO);
    if (quadVBO) glDeleteBuffers(1, &quadVBO);

    if (starsFBO) glDeleteFramebuffers(1, &starsFBO);
    if (starsTex) glDeleteTextures(1, &starsTex);
    if (starsRBO) glDeleteRenderbuffers(1, &starsRBO);

    glDeleteProgram(progGrid);
    glDeleteProgram(progPoints);
    glDeleteProgram(progStar);
    glDeleteProgram(progWarp);

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
