// main.cpp
// OpenGL 3.3 core — Grid (Schwarzschild approx), pixelated black hole, two blurred stars.
// Replace your main.cpp with this file and compile (link GLFW, GLEW, glm).

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

using namespace glm;
using namespace std;

// -------------------------------
// Simple shader helper
// -------------------------------
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint logLen; glGetShaderiv(s, GL_INFO_LOG_LENGTH, &logLen);
        string log(logLen, '\0'); glGetShaderInfoLog(s, logLen, nullptr, &log[0]);
        cerr << "Shader compile error: " << log << endl;
    }
    return s;
}
static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint logLen; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &logLen);
        string log(logLen, '\0'); glGetProgramInfoLog(p, logLen, nullptr, &log[0]);
        cerr << "Program link error: " << log << endl;
    }
    return p;
}

// -------------------------------
// Globals: camera, window size
// -------------------------------
int WIN_W = 800, WIN_H = 600;
struct Camera {
    float radius = 3.5f;
    float azimuth = 0.0f;
    float elevation = glm::pi<float>() / 2.0f;
    float orbitSpeed = 0.005f;
    float zoomSpeed = 0.2f;
    bool dragging = false;
    double lastX = 0, lastY = 0;
    vec3 target = vec3(0.0f, 0.0f, 0.0f);

    vec3 position() const {
        float ele = clamp(elevation, 0.01f, glm::pi<float>() - 0.01f);
        return vec3(
            radius * sin(ele) * cos(azimuth),
            radius * cos(ele),
            radius * sin(ele) * sin(azimuth)
        );
    }
} camera;

// callbacks
void mouse_button_cb(GLFWwindow* w, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        camera.dragging = (action == GLFW_PRESS);
        if (camera.dragging) glfwGetCursorPos(w, &camera.lastX, &camera.lastY);
    }
}
void cursor_pos_cb(GLFWwindow* w, double x, double y) {
    if (camera.dragging) {
        float dx = float(x - camera.lastX);
        float dy = float(y - camera.lastY);
        camera.azimuth += dx * camera.orbitSpeed;
        camera.elevation -= dy * camera.orbitSpeed;
        camera.lastX = x; camera.lastY = y;
    }
}
void scroll_cb(GLFWwindow* w, double xoff, double yoff) {
    camera.radius -= float(yoff) * camera.zoomSpeed;
    camera.radius = clamp(camera.radius, 0.5f, 50.0f);
}
void key_cb(GLFWwindow* w, int key, int sc, int act, int mods) {
    if (key == GLFW_KEY_ESCAPE && act==GLFW_PRESS) glfwSetWindowShouldClose(w, true);
}

// -------------------------------
// Grid generation (indices and vertices)
// -------------------------------
struct GridMesh {
    vector<vec3> verts;
    vector<unsigned int> indices;
    GLuint vao=0, vbo=0, ebo=0;
    int gridSize = 0;
};

float schwarzschildY_vis(float r, float massScale) {
    // Visualization-friendly Schwarzschild-like profile.
    // massScale bigger -> more intense dip.
    float eps = 0.001f;
    // Use a decaying -A/(r+eps) + gaussian bump style to make it visually interesting.
    float A = 0.45f * massScale;
    return -A / (r + 0.08f) * exp(-r*0.6f);
}

void makeGrid(GridMesh &g, int gridSize = 25, float spacing = 0.12f, float massScale = 2.0f) {
    g.gridSize = gridSize;
    g.verts.clear();
    g.indices.clear();
    int N = gridSize*2 + 1;
    for (int z=-gridSize; z<=gridSize; ++z) {
        for (int x=-gridSize; x<=gridSize; ++x) {
            float fx = x * spacing;
            float fz = z * spacing;
            float r = sqrt(fx*fx + fz*fz);
            float fy = schwarzschildY_vis(r, massScale);
            g.verts.emplace_back(fx, fy, fz);
        }
    }
    // indices for lines (horizontal and vertical)
    for (int z=0; z<N; ++z) {
        for (int x=0; x<N; ++x) {
            int i = z*N + x;
            if (x < N-1) { g.indices.push_back(i); g.indices.push_back(i+1); }
            if (z < N-1) { g.indices.push_back(i); g.indices.push_back(i+N); }
        }
    }
    // upload to GPU (VAO / VBO / EBO)
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

// -------------------------------
// Black hole pixels (point grid centered at origin, transformed in vertex shader)
// -------------------------------
struct Pixel {
    float x,y; // local 2D coords in billboard
    vec3 color;
};

struct PixelMesh {
    vector<Pixel> pixels;
    GLuint vao=0, vbo=0;
    int count=0;
};

void makeBlackHolePixels(PixelMesh &pm, int res = 64, float radius = 0.6f) {
    pm.pixels.clear();
    pm.count = 0;
    // Produce regular grid of pixels inside circle (so shape is neat, not random)
    for (int j=0;j<res;++j) {
        for (int i=0;i<res;++i) {
            float u = (i + 0.5f)/res*2.0f - 1.0f; // -1..1
            float v = (j + 0.5f)/res*2.0f - 1.0f;
            float rx = u * radius;
            float ry = v * radius;
            float r = sqrt(rx*rx + ry*ry);
            if (r <= radius) {
                Pixel p;
                p.x = rx;
                p.y = ry;
                // color: center black, inner warm ring, outer bright ring
                float t = r / radius;
                if (t < 0.55f) p.color = vec3(0.0f); // darkest interior
                else if (t < 0.75f) p.color = mix(vec3(0.05f,0.02f,0.01f), vec3(0.9f,0.25f,0.05f), (t-0.55f)/0.2f);
                else p.color = mix(vec3(0.9f,0.25f,0.05f), vec3(1.0f,0.8f,0.2f), (t-0.75f)/0.25f);
                pm.pixels.push_back(p);
            }
        }
    }
    pm.count = (int)pm.pixels.size();
    // upload
    if (!pm.vao) glGenVertexArrays(1, &pm.vao);
    if (!pm.vbo) glGenBuffers(1, &pm.vbo);
    glBindVertexArray(pm.vao);
    glBindBuffer(GL_ARRAY_BUFFER, pm.vbo);
    glBufferData(GL_ARRAY_BUFFER, pm.pixels.size()*sizeof(Pixel), pm.pixels.data(), GL_STATIC_DRAW);
    // layout: location 0 = vec2 pos, location 1 = vec3 color
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(Pixel),(void*)offsetof(Pixel,x));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Pixel),(void*)offsetof(Pixel,color));
    glBindVertexArray(0);
}

// -------------------------------
// Stars: simple point sprites with gaussian in fragment shader
// -------------------------------
struct Star {
    vec3 pos;
    vec3 color;
    float size;
};

struct StarMesh {
    vector<Star> stars;
    GLuint vao=0, vbo=0;
    int count=0;
};

void makeStars(StarMesh &sm) {
    sm.stars.clear();
    // two distant stars
    sm.stars.push_back({ vec3( 6.5f, 0.9f, -10.5f), vec3(1.0f,0.18f,0.08f), 48.0f });
    sm.stars.push_back({ vec3(-7.4f, 0.6f, -11.0f), vec3(1.0f,0.92f,0.35f), 44.0f });
    sm.count = (int)sm.stars.size();
    if (!sm.vao) glGenVertexArrays(1, &sm.vao);
    if (!sm.vbo) glGenBuffers(1, &sm.vbo);

    glBindVertexArray(sm.vao);
    glBindBuffer(GL_ARRAY_BUFFER, sm.vbo);
    glBufferData(GL_ARRAY_BUFFER, sm.stars.size()*sizeof(Star), sm.stars.data(), GL_STATIC_DRAW);
    // layout0: vec3 pos (offset 0), layout1: vec3 color (offset 12), layout2: float size (offset 24)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Star),(void*)offsetof(Star,pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Star),(void*)offsetof(Star,color));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2,1,GL_FLOAT,GL_FALSE,sizeof(Star),(void*)offsetof(Star,size));
    glBindVertexArray(0);
}

// -------------------------------
// Shader sources
// -------------------------------
const char* vs_basic = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main(){
    gl_Position = uMVP * vec4(aPos,1.0);
}
)GLSL";

const char* fs_grid = R"GLSL(
#version 330 core
out vec4 FragColor;
void main(){ FragColor = vec4(1.0); }
)GLSL";

// Black hole pixel shader
const char* vs_pixels = R"GLSL(
#version 330 core
layout(location=0) in vec2 aLocal;   // x,y in billboard local coords
layout(location=1) in vec3 aColor;
uniform mat4 uMVP;        // MVP that already encodes billboard basis + position + scale
uniform float uPointSize;
out vec3 vColor;
void main(){
    // we want each pixel as a point; position is aLocal.xy -> put into vec4 with z=0
    gl_Position = uMVP * vec4(aLocal.x, aLocal.y, 0.0, 1.0);
    vColor = aColor;
    // point size in pixels (can be mapped by scale)
    gl_PointSize = uPointSize;
}
)GLSL";

const char* fs_pixels = R"GLSL(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main(){
    // draw square pixel (no round shape): use full color
    FragColor = vec4(vColor, 1.0);
}
)GLSL";

// Stars: point sprite with gaussian blur
const char* vs_star = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
layout(location=2) in float aSize;
uniform mat4 uMVP;
out vec3 vColor;
out float vSize;
void main(){
    gl_Position = uMVP * vec4(aPos,1.0);
    vColor = aColor;
    vSize = aSize;
    gl_PointSize = aSize;
}
)GLSL";

const char* fs_star = R"GLSL(
#version 330 core
in vec3 vColor;
in float vSize;
out vec4 FragColor;
void main(){
    // gl_PointCoord in 0..1 across the point
    vec2 uv = gl_PointCoord - vec2(0.5);
    float r2 = dot(uv,uv);
    // gaussian falloff tuned to look good
    float sigma = 0.25; // smaller -> tighter star
    float intensity = exp(-r2 / (2.0 * sigma * sigma));
    intensity = clamp(intensity, 0.0, 1.0);
    vec3 col = vColor * (0.5 + 1.1 * intensity);
    FragColor = vec4(col, intensity); // alpha = intensity for blending
}
)GLSL";

// -------------------------------
// Utility: make billboard MVP
// Given camera basis (right, up, look) and object position, build a model matrix that places
// circle plane facing camera; then multiply by scale.
// -------------------------------
mat4 makeBillboardMVP(const vec3 &objPos, const vec3 &camPos, const mat4 &VP, float scale) {
    vec3 look = normalize(camPos - objPos);
    vec3 right = normalize(cross(vec3(0,1,0), look));
    // handle degenerate case when look is parallel to up
    if (length(right) < 1e-4) right = normalize(cross(vec3(1,0,0), look));
    vec3 up = cross(look, right);

    mat4 model(1.0f);
    model[0] = vec4(right * scale, 0.0f);
    model[1] = vec4(up    * scale, 0.0f);
    model[2] = vec4(look  * scale, 0.0f);
    model[3] = vec4(objPos, 1.0f);

    return VP * model;
}

// -------------------------------
// Main
// -------------------------------
int main(){
    srand((unsigned)time(nullptr));
    if (!glfwInit()) { cerr<<"GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(WIN_W, WIN_H, "Black Hole - pixelated", nullptr, nullptr);
    if (!win) { cerr<<"Window creation failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { cerr<<"GLEW init failed\n"; glfwTerminate(); return -1; }

    glfwSetMouseButtonCallback(win, mouse_button_cb);
    glfwSetCursorPosCallback(win, cursor_pos_cb);
    glfwSetScrollCallback(win, scroll_cb);
    glfwSetKeyCallback(win, key_cb);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE); // allow setting gl_PointSize in vertex shader

    // --- make meshes ---
    GridMesh grid; makeGrid(grid, 25, 0.12f, 3.0f); // massScale=3 -> stronger dip

    PixelMesh ph; makeBlackHolePixels(ph, 64, 0.55f); // resolution, radius

    StarMesh stars; makeStars(stars);

    // --- compile shaders & programs ---
    GLuint vs_b = compileShader(GL_VERTEX_SHADER, vs_basic);
    GLuint fs_g = compileShader(GL_FRAGMENT_SHADER, fs_grid);
    GLuint progGrid = linkProgram(vs_b, fs_g);

    GLuint vs_p = compileShader(GL_VERTEX_SHADER, vs_pixels);
    GLuint fs_p = compileShader(GL_FRAGMENT_SHADER, fs_pixels);
    GLuint progPixels = linkProgram(vs_p, fs_p);

    GLuint vs_s = compileShader(GL_VERTEX_SHADER, vs_star);
    GLuint fs_s = compileShader(GL_FRAGMENT_SHADER, fs_star);
    GLuint progStar = linkProgram(vs_s, fs_s);

    // projection matrix (fixed)
    mat4 proj = perspective(radians(60.0f), float(WIN_W)/float(WIN_H), 0.1f, 200.0f);

    // black hole position (slightly lower so it sits in grid)
    vec3 blackPos = vec3(0.0f, -0.35f, 0.0f);

    // main loop
    while (!glfwWindowShouldClose(win)) {
        glViewport(0,0,WIN_W,WIN_H);
        glClearColor(0.0f,0.0f,0.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mat4 view = lookAt(camera.position(), camera.target, vec3(0,1,0));
        mat4 VP = proj * view;
        vec3 camPos = camera.position();

        // -- draw grid (lines) --
        glUseProgram(progGrid);
        GLint loc = glGetUniformLocation(progGrid, "uMVP");
        if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, value_ptr(VP));
        glBindVertexArray(grid.vao);
        glDrawElements(GL_LINES, (GLsizei)grid.indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // -- draw accretion disk (simple triangles using same approach as old code) --
        // We'll draw a ring using same billboard technique but with triangles precomputed in CPU:
        // For simplicity re-use the pixel shader program but draw nothing heavy here; keep disk as simple colored ring.
        // (To keep code short: use pixels as pseudo-disk fill behind black hole)
        // -- draw pixelated black hole (each point is a pixel) --
        glUseProgram(progPixels);
        // compute billboard MVP so local aLocal.x/y are in correct basis and scale
        float bhScale = 0.6f; // scale to fit visually
        mat4 bhMVP = makeBillboardMVP(blackPos, camPos, VP, bhScale);
        GLint loc2 = glGetUniformLocation(progPixels, "uMVP");
        if (loc2 >= 0) glUniformMatrix4fv(loc2, 1, GL_FALSE, value_ptr(bhMVP));
        // point size (in pixels) — tune so "blocks" are visible regardless of resolution
        float pixelSize = 6.0f; // change for larger/smaller blocks
        GLint locPS = glGetUniformLocation(progPixels, "uPointSize");
        if (locPS >= 0) glUniform1f(locPS, pixelSize);

        glBindVertexArray(ph.vao);
        glDrawArrays(GL_POINTS, 0, ph.count);
        glBindVertexArray(0);

        // -- draw stars (point sprites with gaussian blur) --
        glUseProgram(progStar);
        GLint locS = glGetUniformLocation(progStar, "uMVP");
        if (locS >= 0) glUniformMatrix4fv(locS, 1, GL_FALSE, value_ptr(VP));
        glBindVertexArray(stars.vao);
        // each star is one vertex: draw arrays with count
        glDrawArrays(GL_POINTS, 0, stars.count);
        glBindVertexArray(0);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    // cleanup
    glDeleteProgram(progGrid);
    glDeleteProgram(progPixels);
    glDeleteProgram(progStar);
    if (grid.vao) glDeleteVertexArrays(1, &grid.vao);
    if (grid.vbo) glDeleteBuffers(1, &grid.vbo);
    if (grid.ebo) glDeleteBuffers(1, &grid.ebo);
    if (ph.vao) glDeleteVertexArrays(1, &ph.vao);
    if (ph.vbo) glDeleteBuffers(1, &ph.vbo);
    if (stars.vao) glDeleteVertexArrays(1, &stars.vao);
    if (stars.vbo) glDeleteBuffers(1, &stars.vbo);

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
