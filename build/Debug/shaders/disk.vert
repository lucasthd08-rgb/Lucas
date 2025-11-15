#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
using namespace glm;

// ================= Camera =================
struct Camera {
    vec3 target = vec3(0.0f);
    float radius = 3.5f;
    float azimuth = 0.0f;
    float elevation = glm::pi<float>() / 2.0f;
    float orbitSpeed = 0.005f;
    float zoomSpeed = 0.2f;
    bool dragging = false;
    double lastX = 0.0, lastY = 0.0;

    vec3 position() const {
        float ele = clamp(elevation, 0.01f, glm::pi<float>() - 0.01f);
        return vec3(
            radius * sin(ele) * cos(azimuth),
            radius * cos(ele),
            radius * sin(ele) * sin(azimuth)
        );
    }
} camera;

// ================= Callbacks =================
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods){
    if(button == GLFW_MOUSE_BUTTON_LEFT){
        camera.dragging = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &camera.lastX, &camera.lastY);
    }
}
void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos){
    if(camera.dragging){
        float dx = float(xpos - camera.lastX);
        float dy = float(ypos - camera.lastY);
        camera.azimuth += dx * camera.orbitSpeed;
        camera.elevation -= dy * camera.orbitSpeed;
        camera.lastX = xpos;
        camera.lastY = ypos;
    }
}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
    camera.radius -= float(yoffset) * camera.zoomSpeed;
    camera.radius = clamp(camera.radius, 0.5f, 10.0f);
}

// ================= Grid (Schwarzschild) =================
vector<vec3> gridVerts;
vector<GLuint> gridIdx;
GLuint gridVAO = 0, gridVBO = 0, gridEBO = 0;

// --- Schwarzschild curvature approximation ---
float schwarzschildY(float r) {
    float eps = 0.01f;
    float M = 1.0f; // aumentar massa para curvatura mais forte
    return -M / (r + eps);
}

void generateGrid(int gridSize = 25, float spacing = 0.12f){
    gridVerts.clear();
    gridIdx.clear();

    for(int z = 0; z <= gridSize; ++z){
        for(int x = 0; x <= gridSize; ++x){
            float wx = (x - gridSize/2) * spacing;
            float wz = (z - gridSize/2) * spacing;
            float r = sqrt(wx*wx + wz*wz);
            float y = schwarzschildY(r);
            gridVerts.emplace_back(wx, y, wz);
        }
    }
    for(int z = 0; z < gridSize; ++z){
        for(int x = 0; x < gridSize; ++x){
            int i = z*(gridSize+1) + x;
            gridIdx.push_back(i); gridIdx.push_back(i+1);
            gridIdx.push_back(i); gridIdx.push_back(i + (gridSize+1));
        }
    }

    if(gridVAO == 0) glGenVertexArrays(1, &gridVAO);
    if(gridVBO == 0) glGenBuffers(1, &gridVBO);
    if(gridEBO == 0) glGenBuffers(1, &gridEBO);

    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, gridVerts.size()*sizeof(vec3), gridVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, gridIdx.size()*sizeof(GLuint), gridIdx.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    glBindVertexArray(0);
}

// ================= Shader util =================
GLuint compileShaderSrc(const char* vertSrc, const char* fragSrc){
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vertSrc, nullptr);
    glCompileShader(v);
    GLint ok;
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    if(!ok){ char buf[1024]; glGetShaderInfoLog(v,1024,nullptr,buf); cerr<<"Vert shader error: "<<buf<<endl;}

    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fragSrc, nullptr);
    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if(!ok){ char buf[1024]; glGetShaderInfoLog(f,1024,nullptr,buf); cerr<<"Frag shader error: "<<buf<<endl;}

    GLuint p = glCreateProgram();
    glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){ char buf[1024]; glGetProgramInfoLog(p,1024,nullptr,buf); cerr<<"Program link error: "<<buf<<endl;}
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ================= Geometry =================
vector<vec3> generateCircle(float r=0.12f, int segs=64){
    vector<vec3> verts;
    verts.emplace_back(0,0,0);
    for(int i=0;i<=segs;i++){
        float a = float(i)/segs * glm::two_pi<float>();
        verts.emplace_back(cos(a)*r, sin(a)*r,0);
    }
    return verts;
}

vector<vec3> generateDisk(float innerR=0.28f, float outerR=0.6f, int segs=128){
    vector<vec3> verts;
    for(int i=0;i<segs;i++){
        float a0 = float(i)/segs * glm::two_pi<float>();
        float a1 = float(i+1)/segs * glm::two_pi<float>();
        float r0 = innerR, r1 = outerR;
        float y0 = -0.02f + 0.03f * exp(-r0*4.5f); // levemente baixo para coincidir com circle
        float y1 = -0.02f + 0.03f * exp(-r1*4.5f);
        verts.emplace_back(cos(a0)*r0, y0, sin(a0)*r0);
        verts.emplace_back(cos(a0)*r1, y1, sin(a0)*r1);
        verts.emplace_back(cos(a1)*r0, y0, sin(a1)*r0);
        verts.emplace_back(cos(a1)*r0, y0, sin(a1)*r0);
        verts.emplace_back(cos(a0)*r1, y1, sin(a0)*r1);
        verts.emplace_back(cos(a1)*r1, y1, sin(a1)*r1);
    }
    return verts;
}

// ================= Star struct =================
struct Star {
    vec3 position;
    vec3 color;
    GLuint VAO, VBO;
};

// ================= Shaders =================

// basic vertex with uMVP (same style as seu código que funcionou)
const char* vertBasic = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
out vec2 localPos;
void main(){
    // pass local 2D position (x,y) for the star shader falloff (circle geometry is centered on 0)
    localPos = aPos.xy;
    gl_Position = uMVP * vec4(aPos,1.0);
}
)";

// simple white fragment (grid uses its own color in CPU here)
const char* fragWhite = R"(
#version 330 core
out vec4 FragColor;
void main(){ FragColor = vec4(1.0); }
)";

// black hole fragment (solid black)
const char* blackFrag = R"(
#version 330 core
out vec4 FragColor;
void main(){ FragColor = vec4(0.0,0.0,0.0,1.0); }
)";

// disk fragment (keeps orange)
const char* diskFrag = R"(
#version 330 core
out vec4 FragColor;
void main(){ FragColor = vec4(1.0,0.75,0.25,1.0); }
)";

// star fragment: uses localPos to compute radial falloff (gives smooth blur)
const char* starFrag = R"(
#version 330 core
in vec2 localPos;
out vec4 FragColor;
uniform vec3 starColor;
uniform float intensity; // multiplier
void main(){
    float d = length(localPos);
    // smooth Gaussian-like falloff
    float fall = exp(-12.0 * d * d);
    // clamp to avoid black edges
    fall = clamp(fall, 0.0, 1.0);
    vec3 col = starColor * (0.6 + 0.8 * fall) * intensity;
    // additive-like result but keep alpha 1 for proper blending order
    FragColor = vec4(col, fall);
}
)";

// ================= Main =================
int main(){
    if(!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800,600,"Black Hole Simulation",nullptr,nullptr);
    if(!window){ glfwTerminate(); return -1;}
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if(glewInit()!=GLEW_OK){ cerr<<"GLEW failed\n"; return -1;}

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLuint gridShader = compileShaderSrc(vertBasic, fragWhite);
    generateGrid();

    // Buraco negro (circle)
    vector<vec3> circleVerts = generateCircle(0.15f);
    GLuint blackHoleShader = compileShaderSrc(vertBasic, blackFrag);

    GLuint blackVAO, blackVBO;
    glGenVertexArrays(1,&blackVAO);
    glGenBuffers(1,&blackVBO);
    glBindVertexArray(blackVAO);
    glBindBuffer(GL_ARRAY_BUFFER,blackVBO);
    glBufferData(GL_ARRAY_BUFFER,circleVerts.size()*sizeof(vec3),circleVerts.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    glBindVertexArray(0);

    // Disco de acreção
    vector<vec3> diskVerts = generateDisk();
    GLuint diskShaderProg = compileShaderSrc(vertBasic, diskFrag);
    GLuint diskVAO, diskVBO;
    glGenVertexArrays(1,&diskVAO);
    glGenBuffers(1,&diskVBO);
    glBindVertexArray(diskVAO);
    glBindBuffer(GL_ARRAY_BUFFER,diskVBO);
    glBufferData(GL_ARRAY_BUFFER,diskVerts.size()*sizeof(vec3),diskVerts.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    glBindVertexArray(0);

    // Estrelas (coloridas e com blur real no fragment shader)
    GLuint starShader = compileShaderSrc(vertBasic, starFrag);
    vector<Star> stars = {
        { vec3( 2.0f,  0.15f,  2.5f), vec3(1.0f, 0.9f, 0.3f), 0, 0 }, // amarela distante (mais brilhante)
        { vec3(-2.4f,  0.12f, -2.2f), vec3(1.0f, 0.15f, 0.15f), 0, 0 }  // vermelha distante
    };

    // reuse circle geometry for star billboards (centered mesh)
    for(auto &s : stars){
        glGenVertexArrays(1, &s.VAO);
        glGenBuffers(1, &s.VBO);
        glBindVertexArray(s.VAO);
        glBindBuffer(GL_ARRAY_BUFFER, s.VBO);
        glBufferData(GL_ARRAY_BUFFER, circleVerts.size()*sizeof(vec3), circleVerts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
        glBindVertexArray(0);
    }

    // === Loop ===
    while(!glfwWindowShouldClose(window)){
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mat4 view = lookAt(camera.position(), camera.target, vec3(0,1,0));
        mat4 proj = perspective(radians(60.0f),800.0f/600.0f,0.1f,100.0f);
        mat4 VP = proj * view;

        // Grid
        glUseProgram(gridShader);
        glUniformMatrix4fv(glGetUniformLocation(gridShader,"uMVP"),1,GL_FALSE,value_ptr(VP));
        glBindVertexArray(gridVAO);
        glDrawElements(GL_LINES, gridIdx.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // Disco (note: y set slightly negative in generator so it sits around circle)
        glUseProgram(diskShaderProg);
        glUniformMatrix4fv(glGetUniformLocation(diskShaderProg,"uMVP"),1,GL_FALSE,value_ptr(VP));
        glBindVertexArray(diskVAO);
        glDrawArrays(GL_TRIANGLES,0, (GLsizei)diskVerts.size());
        glBindVertexArray(0);

        // Buraco negro (olhando pra câmera), abaixado levemente (y = -0.05)
        vec3 objPos = camera.target + vec3(0.0f, -0.1f, 0.0f); // abaixado para alinhar com disco/grid
        vec3 camPos = camera.position();
        vec3 look = normalize(camPos - objPos);
        vec3 right = normalize(cross(vec3(0,1,0), look));
        vec3 up = cross(look, right);
        mat4 model = mat4(1.0f);
        model[0] = vec4(right, 0.0f);
        model[1] = vec4(up, 0.0f);
        model[2] = vec4(look, 0.0f);
        model[3] = vec4(objPos,0.5f);
        mat4 MVP = VP * model;

        glUseProgram(blackHoleShader);
        glUniformMatrix4fv(glGetUniformLocation(blackHoleShader,"uMVP"),1,GL_FALSE,value_ptr(MVP));
        glBindVertexArray(blackVAO);
        glDrawArrays(GL_TRIANGLE_FAN,0,(GLsizei)circleVerts.size());
        glBindVertexArray(0);

        // Estrelas
        // torná-las maiores/brilhantes: ajustar intensidade e escala do model
        for(size_t i=0;i<stars.size();++i){
            auto &s = stars[i];
            vec3 toCam = normalize(camPos - s.position);
            vec3 r = normalize(cross(vec3(0,1,0), toCam));
            vec3 u = cross(toCam, r);
            float starSize = (i==0) ? 0.45f : 0.36f; // primeira estrela maior
            mat4 modelS = mat4(1.0f);
            modelS[0] = vec4(r*starSize,0.0f);
            modelS[1] = vec4(u*starSize,0.0f);
            modelS[2] = vec4(toCam*starSize*0.02f,0.0f); // pequena profundidade
            modelS[3] = vec4(s.position,1.0f);
            mat4 mvpS = VP * modelS;

            glUseProgram(starShader);
            glUniformMatrix4fv(glGetUniformLocation(starShader,"uMVP"),1,GL_FALSE,value_ptr(mvpS));
            // intensity: aproximação de brilho (dependendo da distância você pode ajustar mais)
            float intensity = 1.6f;
            glUniform1f(glGetUniformLocation(starShader,"intensity"), intensity);
            glUniform3fv(glGetUniformLocation(starShader,"starColor"),1,value_ptr(s.color));

            glBindVertexArray(s.VAO);
            glDrawArrays(GL_TRIANGLE_FAN,0,(GLsizei)circleVerts.size());
            glBindVertexArray(0);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // cleanup
    glfwTerminate();
    return 0;
}
