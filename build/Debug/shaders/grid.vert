// main.cpp
// Pixelated BH + horizontal pixel accretion disk + visible photon ring (billboard) + blurred stars.
// Fixes: disk uses BH vertical position (so not floating above) and photon ring rendered as billboard
// Compile example: g++ main.cpp -lglfw -lGLEW -lGL -ldl -o blackhole

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

// Window
const int WIN_W = 800;
const int WIN_H = 600;

// Scene parameters (tweakable)
int BH_PIXEL_RES = 128;
float BH_RADIUS = 0.65f;
float pixelPointSize = 6.0f;

// Disk parameters (horizontal plane, centered at blackPos.y)
int DISK_RADIAL_STEPS = 36;
int DISK_ANGULAR_STEPS = 360;
float DISK_INNER = 0.50f;
float DISK_OUTER = 0.95f;
float DISK_THICKNESS = 0.04f;

// Photon ring billboard radii (in billboard local units)
float PH_RING_IN = BH_RADIUS * 0.8;
float PH_RING_OUT = BH_RADIUS * 0.9f;
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
    vec3 target = vec3(0.0f, 0.0f, 0.0f);
    vec3 position() const {
        float ele = clamp(elevation, 0.01f, glm::pi<float>() - 0.01f);
        return vec3(radius * sin(ele) * cos(azimuth),
                    radius * cos(ele),
                    radius * sin(ele) * sin(azimuth));
    }
} camera;

bool autoRotate = false;

// Helpers
static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok){
        GLint logLen; glGetShaderiv(s, GL_INFO_LOG_LENGTH, &logLen);
        if(logLen>0){
            string log(logLen, '\0');
            glGetShaderInfoLog(s, logLen, nullptr, &log[0]);
            cerr<<"Shader compile error:\n"<<log<<endl;
        } else cerr<<"Shader compile failed (no log)\n";
    }
    return s;
}
static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){
        GLint logLen; glGetProgramiv(p, GL_INFO_LOG_LENGTH, &logLen);
        if(logLen>0){
            string log(logLen, '\0');
            glGetProgramInfoLog(p, logLen, nullptr, &log[0]);
            cerr<<"Program link error:\n"<<log<<endl;
        } else cerr<<"Program link failed (no log)\n";
    }
    return p;
}

// Mesh containers
struct Pixel {
    float x,y,z;
    float r,g,b,a;
};
struct MeshBuffer {
    vector<Pixel> pixels;
    GLuint vao=0, vbo=0;
    int count=0;
};

// Shaders
const char* vs_points = R"glsl(
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
)glsl";

const char* fs_points = R"glsl(
#version 330 core
in vec4 vCol;
out vec4 FragColor;
void main(){
    // square pixel
    FragColor = vCol;
}
)glsl";

const char* vs_star = R"glsl(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aCol;
layout(location=2) in float aSize;
uniform mat4 uMVP;
out vec3 vCol;
out float vSize;
void main(){
    vCol = aCol;
    vSize = aSize;
    gl_Position = uMVP * vec4(aPos,1.0);
    gl_PointSize = aSize;
}
)glsl";

const char* fs_star = R"glsl(
#version 330 core
in vec3 vCol;
in float vSize;
out vec4 FragColor;
void main(){
    vec2 uv = gl_PointCoord - vec2(0.5);
    float r2 = dot(uv,uv);
    float sigma = 0.28;
    float intensity = exp(-r2 / (2.0 * sigma * sigma));
    intensity = clamp(intensity, 0.0, 1.0);
    vec3 col = vCol * (0.6 + 1.2*intensity);
    FragColor = vec4(col, intensity);
}
)glsl";

const char* vs_grid = R"glsl(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main(){ gl_Position = uMVP * vec4(aPos,1.0); }
)glsl";
const char* fs_grid = R"glsl(
#version 330 core
out vec4 FragColor;
void main(){ FragColor = vec4(0.95,0.7,0.45,1.0); }
)glsl";

// Billboard model for ring/bh
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

// Generate BH interior pixels (billboard local)
void generateBlackHolePixels(MeshBuffer &mb, int res, float radius){
    mb.pixels.clear();
    for(int j=0;j<res;++j){
        for(int i=0;i<res;++i){
            float u = (i + 0.5f)/float(res)*2.0f - 1.0f;
            float v = (j + 0.5f)/float(res)*2.0f - 1.0f;
            float x = u * radius;
            float y = v * radius;
            float r = sqrt(x*x + y*y);
            if(r <= radius){
                Pixel p;
                p.x = x; p.y = y; p.z = 0.0f;
                p.r = 0.0f; p.g = 0.0f; p.b = 0.0f; p.a = 1.0f;
                mb.pixels.push_back(p);
            }
        }
    }
    mb.count = (int)mb.pixels.size();
}

// Generate horizontal disk pixels in world coords, centered at (0, blackY, 0)
void generateDiskPixelsWorld(MeshBuffer &mb, float innerR, float outerR, float thickness,
                             int radialSteps, int angularSteps, float blackY)
{
    mb.pixels.clear();
    for(int ri=0; ri<radialSteps; ++ri){
        float t = (ri+0.5f)/float(radialSteps);
        float r = mix(innerR, outerR, t);
        int aSteps = angularSteps;
        // optionally vary angular density by radial position if desired
        for(int ai=0; ai<aSteps; ++ai){
            float a = (ai + 0.5f)/float(aSteps) * 2.0f * M_PI;
            float jitter = ((rand()%1000)/1000.0f - 0.5f) * 0.003f;
            float rr = r + jitter;
            // sample a few y positions for mild thickness
            for(int yi=0; yi<3; ++yi){
                float y = blackY + (-thickness*0.35f + yi * (thickness*0.35f));
                Pixel p;
                p.x = cos(a)*rr;
                p.z = sin(a)*rr;
                p.y = y;
                float radialNorm = smoothstep(innerR, outerR, rr);
                vec3 innerColor = vec3(0.96f, 0.12f, 0.03f); // red
                vec3 outerColor = vec3(1.0f, 0.78f, 0.18f); // orange
                vec3 col = mix(innerColor, outerColor, radialNorm);
                // alpha fade: slightly stronger near outer edge
                float alpha = 0.85f * (0.6f + 0.6f*radialNorm);
                p.r = col.r; p.g = col.g; p.b = col.b; p.a = alpha;
                mb.pixels.push_back(p);
            }
        }
    }
    mb.count = (int)mb.pixels.size();
}

// Generate photon ring as billboard-local pixels (so it always faces camera)
void generatePhotonRingBillboard(MeshBuffer &mb, float inR, float outR, int samples){
    mb.pixels.clear();
    float radialStep = (outR - inR) / 4.0f;
    for(int s=0; s<samples; ++s){
        float a = (s + 0.5f)/float(samples) * 2.0f * M_PI;
        for(float r = inR; r <= outR; r += radialStep){
            float jitter = ((rand()%1000)/1000.0f - 0.5f) * 0.004f;
            float rr = r + jitter;
            Pixel p;
            p.x =  cos(a) * rr;
            p.y = sin(a) * rr;
            p.z = 0.0f;
            // color gradient orange->red outward->inward
            float tt = (rr - inR) / (outR - inR);
            vec3 col = mix(vec3(1.0f, 0.6f, 0.1f), vec3(1.0f, 0.14f, 0.02f), tt);
            float alpha = 0.95f * (0.5f + 0.8f * (1.0f - abs(tt-0.5f)));
            p.r = col.r; p.g = col.g; p.b = col.b; p.a = alpha;
            mb.pixels.push_back(p);
        }
    }
    mb.count = (int)mb.pixels.size();
}

// Stars
void setupStars(){
    stars.clear();
    stars.push_back({ vec3(-3.8f, 0.9f, -8.6f), vec3(1.0f, 0.66f, 0.32f), 84.0f });
    stars.push_back({ vec3(4.2f, 0.7f, -9.3f), vec3(1.0f, 0.18f, 0.08f), 92.0f });
}

// Upload mesh
void uploadMesh(MeshBuffer &mb){
    if(!mb.vao) glGenVertexArrays(1, &mb.vao);
    if(!mb.vbo) glGenBuffers(1, &mb.vbo);
    glBindVertexArray(mb.vao);
    glBindBuffer(GL_ARRAY_BUFFER, mb.vbo);
    glBufferData(GL_ARRAY_BUFFER, mb.pixels.size() * sizeof(Pixel), mb.pixels.data(), GL_STATIC_DRAW);
    // layout 0: vec3 pos, 1: vec4 color
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Pixel), (void*)offsetof(Pixel, x));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Pixel), (void*)offsetof(Pixel, r));
    glBindVertexArray(0);
}

// Stars upload
GLuint starsVAO=0, starsVBO=0;
void uploadStars(){
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

// Grid mesh for background
struct GridMesh { vector<vec3> verts; vector<unsigned int> indices; GLuint vao=0,vbo=0,ebo=0; int indexCount=0; };
void generateGrid(GridMesh &g, int gridSize=28, float spacing=0.12f, float massScale=3.2f){
    g.verts.clear(); g.indices.clear();
    int N = gridSize*2 + 1;
    for(int z=-gridSize; z<=gridSize; ++z){
        for(int x=-gridSize; x<=gridSize; ++x){
            float fx = x * spacing;
            float fz = z * spacing;
            float r = sqrt(fx*fx + fz*fz);
            float A = 0.45f * massScale;
            float fy = -A / (r + 0.08f) * exp(-r*0.6f);
            g.verts.emplace_back(fx, fy - 0.28f, fz);
        }
    }
    for(int z=0; z<N; ++z){
        for(int x=0; x<N; ++x){
            int i = z*N + x;
            if(x < N-1){ g.indices.push_back(i); g.indices.push_back(i+1); }
            if(z < N-1){ g.indices.push_back(i); g.indices.push_back(i+N); }
        }
    }
    g.indexCount = (int)g.indices.size();
    if(!g.vao) glGenVertexArrays(1, &g.vao);
    if(!g.vbo) glGenBuffers(1, &g.vbo);
    if(!g.ebo) glGenBuffers(1, &g.ebo);
    glBindVertexArray(g.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g.vbo);
    glBufferData(GL_ARRAY_BUFFER, g.verts.size()*sizeof(vec3), g.verts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, g.indices.size()*sizeof(unsigned int), g.indices.data(), GL_STATIC_DRAW);
    glBindVertexArray(0);
}

// Callbacks
void mouse_button_cb(GLFWwindow* w, int button, int action, int mods){
    if(button==GLFW_MOUSE_BUTTON_LEFT){
        camera.dragging = (action==GLFW_PRESS);
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
    if(key==GLFW_KEY_ESCAPE && action==GLFW_PRESS) glfwSetWindowShouldClose(w,true);
    if(key==GLFW_KEY_R && action==GLFW_PRESS) autoRotate = !autoRotate;
    if(key==GLFW_KEY_UP && (action==GLFW_PRESS||action==GLFW_REPEAT)){
        BH_PIXEL_RES = glm::min(1024, BH_PIXEL_RES + 16);
        cerr<<"BH_PIXEL_RES="<<BH_PIXEL_RES<<"\n";
    }
    if(key==GLFW_KEY_DOWN && (action==GLFW_PRESS||action==GLFW_REPEAT)){
        BH_PIXEL_RES = glm::max(16, BH_PIXEL_RES - 16);
        cerr<<"BH_PIXEL_RES="<<BH_PIXEL_RES<<"\n";
    }
    if(key==GLFW_KEY_KP_ADD && (action==GLFW_PRESS||action==GLFW_REPEAT)){
        pixelPointSize = glm::min(64.0f, pixelPointSize + 1.0f);
        cerr<<"pixelPointSize="<<pixelPointSize<<"\n";
    }
    if(key==GLFW_KEY_KP_SUBTRACT && (action==GLFW_PRESS||action==GLFW_REPEAT)){
        pixelPointSize = glm::max(1.0f, pixelPointSize - 1.0f);
        cerr<<"pixelPointSize="<<pixelPointSize<<"\n";
    }
}

int main(){
    srand((unsigned)time(nullptr));
    if(!glfwInit()){ cerr<<"GLFW init failed\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(WIN_W, WIN_H, "Pixel Black Hole - fixed disk & ring", nullptr, nullptr);
    if(!win){ cerr<<"Window creation failed\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK){ cerr<<"GLEW init failed\n"; glfwTerminate(); return -1; }

    glfwSetMouseButtonCallback(win, mouse_button_cb);
    glfwSetCursorPosCallback(win, cursor_pos_cb);
    glfwSetScrollCallback(win, scroll_cb);
    glfwSetKeyCallback(win, key_cb);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Compile programs
    GLuint vsP = compileShader(GL_VERTEX_SHADER, vs_points);
    GLuint fsP = compileShader(GL_FRAGMENT_SHADER, fs_points);
    GLuint progPoints = linkProgram(vsP, fsP);

    GLuint vsS = compileShader(GL_VERTEX_SHADER, vs_star);
    GLuint fsS = compileShader(GL_FRAGMENT_SHADER, fs_star);
    GLuint progStar = linkProgram(vsS, fsS);

    GLuint vsG = compileShader(GL_VERTEX_SHADER, vs_grid);
    GLuint fsG = compileShader(GL_FRAGMENT_SHADER, fs_grid);
    GLuint progGrid = linkProgram(vsG, fsG);

    // Scene geometry
    GridMesh grid; generateGrid(grid, 28, 0.12f, 3.2f);

    // Black hole and disk/ring buffers
    MeshBuffer bhPixels, diskPixels, ringPixels;
    generateBlackHolePixels(bhPixels, BH_PIXEL_RES, BH_RADIUS);

    // black hole vertical position so disk sits around it
    vec3 blackPos = vec3(0.0f, -0.28f, 0.0f);
    // generate disk in world coords using blackPos.y
    generateDiskPixelsWorld(diskPixels, DISK_INNER, DISK_OUTER, DISK_THICKNESS,
                            DISK_RADIAL_STEPS, DISK_ANGULAR_STEPS, blackPos.y);
    // generate photon ring as billboard local pixels
    generatePhotonRingBillboard(ringPixels, PH_RING_IN, PH_RING_OUT, PH_RING_SAMPLES);

    // Stars
    setupStars();
    uploadStars();

    // Upload pixel buffers
    uploadMesh(bhPixels);
    uploadMesh(diskPixels);
    uploadMesh(ringPixels);

    mat4 proj = perspective(radians(60.0f), float(WIN_W)/float(WIN_H), 0.1f, 300.0f);

    GLint loc_uMVP_points = glGetUniformLocation(progPoints, "uMVP");
    GLint loc_pointSize = glGetUniformLocation(progPoints, "uPointSize");
    GLint loc_uMVP_star = glGetUniformLocation(progStar, "uMVP");
    GLint loc_uMVP_grid = glGetUniformLocation(progGrid, "uMVP");

    // Main loop
    double lastT = glfwGetTime();
    while(!glfwWindowShouldClose(win)){
        double now = glfwGetTime();
        double dt = now - lastT;
        lastT = now;

        if(!camera.dragging && autoRotate) camera.azimuth += 0.0009f;

        glViewport(0,0,WIN_W,WIN_H);
        glClearColor(0.02f,0.01f,0.01f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        vec3 camPos = camera.position();
        mat4 view = lookAt(camPos, camera.target, vec3(0,1,0));
        mat4 VP = proj * view;

        // draw grid
        glUseProgram(progGrid);
        if(loc_uMVP_grid>=0) glUniformMatrix4fv(loc_uMVP_grid,1,GL_FALSE,value_ptr(VP));
        glBindVertexArray(grid.vao);
        glDrawElements(GL_LINES, grid.indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // draw disk (world horizontal) first
        glUseProgram(progPoints);
        mat4 diskModel = translate(mat4(1.0f), vec3(0.0f, 0.0f, 0.0f)); // disk pixels already at blackPos.y
        mat4 diskMVP = VP * diskModel;
        if(loc_uMVP_points>=0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(diskMVP));
        if(loc_pointSize>=0) glUniform1f(loc_pointSize, pixelPointSize * 1.05f);
        glBindVertexArray(diskPixels.vao);
        glDrawArrays(GL_POINTS, 0, diskPixels.count);
        glBindVertexArray(0);

        // draw photon ring billboard so it stays facing camera and sits visually just outside BH
        mat4 ringModel = makeBillboardModel(vec3(blackPos.x, blackPos.y + 0.0f, blackPos.z), camPos, 1.0f);
        mat4 ringMVP = VP * ringModel;
        if(loc_uMVP_points>=0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(ringMVP));
        if(loc_pointSize>=0) glUniform1f(loc_pointSize, pixelPointSize * 0.95f);
        glBindVertexArray(ringPixels.vao);
        glDrawArrays(GL_POINTS, 0, ringPixels.count);
        glBindVertexArray(0);

        // draw BH billboard center on top so it occludes disk center
        mat4 bhModel = makeBillboardModel(blackPos, camPos, 0.7f);
        mat4 bhMVP = VP * bhModel;
        if(loc_uMVP_points>=0) glUniformMatrix4fv(loc_uMVP_points, 1, GL_FALSE, value_ptr(bhMVP));
        if(loc_pointSize>=0) glUniform1f(loc_pointSize, pixelPointSize);
        glBindVertexArray(bhPixels.vao);
        glDrawArrays(GL_POINTS, 0, bhPixels.count);
        glBindVertexArray(0);

        // draw stars
        glUseProgram(progStar);
        if(loc_uMVP_star>=0) glUniformMatrix4fv(loc_uMVP_star, 1, GL_FALSE, value_ptr(VP));
        glBindVertexArray(starsVAO);
        glDrawArrays(GL_POINTS, 0, (GLsizei)stars.size());
        glBindVertexArray(0);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    // cleanup
    if(grid.vao) glDeleteVertexArrays(1,&grid.vao);
    if(grid.vbo) glDeleteBuffers(1,&grid.vbo);
    if(grid.ebo) glDeleteBuffers(1,&grid.ebo);

    if(bhPixels.vao) glDeleteVertexArrays(1,&bhPixels.vao);
    if(bhPixels.vbo) glDeleteBuffers(1,&bhPixels.vbo);
    if(diskPixels.vao) glDeleteVertexArrays(1,&diskPixels.vao);
    if(diskPixels.vbo) glDeleteBuffers(1,&diskPixels.vbo);
    if(ringPixels.vao) glDeleteVertexArrays(1,&ringPixels.vao);
    if(ringPixels.vbo) glDeleteBuffers(1,&ringPixels.vbo);

    if(starsVAO) glDeleteVertexArrays(1,&starsVAO);
    if(starsVBO) glDeleteBuffers(1,&starsVBO);

    glDeleteProgram(progPoints);
    glDeleteProgram(progStar);
    glDeleteProgram(progGrid);

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
