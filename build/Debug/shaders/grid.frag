// main.cpp
// Versão atualizada: mantém a estrutura do seu código que funcionou,
// ajusta estrelas (distância, tamanho, blur) e posicionamento do buraco negro.
// Não muda callbacks nem lógica de geração do grid.

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

// ------------------ Camera ------------------
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

// ------------------ Callbacks ------------------
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
    camera.radius = clamp(camera.radius, 0.5f, 20.0f);
}

// ------------------ Grid (Schwarzschild approx) ------------------
vector<vec3> gridVerts;
vector<GLuint> gridIdx;
GLuint gridVAO = 0, gridVBO = 0, gridEBO = 0;

// Retorna deslocamento y (em UNIDADES DE CENA) dado r (em UNIDADES DE CENA).
// Esta versão usa parâmetros de Sagittarius A* e escala para visualização.
// Ajuste `sceneUnitMeters`, `A` e `decayFactor` para controlar a aparência.
float schwarzschildY(float r) {
    float M = 2.0f;
    return -0.35f * exp(-r * 1.2f) - (M / (r*r + 2.0f)); 
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
            // linhas horizontais e verticais (GL_LINES)
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

// ------------------ Shader util ------------------
void printShaderLog(GLuint shader) {
    GLint logLen = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
    if(logLen > 1) {
        std::vector<char> log(logLen);
        glGetShaderInfoLog(shader, logLen, nullptr, log.data());
        cerr << "Shader log:\n" << log.data() << endl;
    }
}
void printProgramLog(GLuint prog) {
    GLint logLen = 0;
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLen);
    if(logLen > 1) {
        std::vector<char> log(logLen);
        glGetProgramInfoLog(prog, logLen, nullptr, log.data());
        cerr << "Program log:\n" << log.data() << endl;
    }
}

GLuint compileShaderSrc(const char* vertSrc, const char* fragSrc){
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vertSrc, nullptr);
    glCompileShader(v);
    GLint ok;
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    if(!ok){ printShaderLog(v); cerr<<"Vert shader compile error\n"; }

    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fragSrc, nullptr);
    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if(!ok){ printShaderLog(f); cerr<<"Frag shader compile error\n"; }

    GLuint p = glCreateProgram();
    glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){ printProgramLog(p); cerr<<"Program link error\n"; }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ------------------ Billboards / Geometry ------------------
vector<vec3> generateCircle(float r=0.12f, int segs=64){
    vector<vec3> verts;
    verts.emplace_back(0,0,0); // center
    for(int i=0;i<=segs;i++){
        float a = float(i)/segs * glm::two_pi<float>();
        verts.emplace_back(cos(a)*r, sin(a)*r,0);
    }
    return verts;
}

vector<vec3> generateDisk(float innerR=0.25f, float outerR=0.55f, int segs=128){
    vector<vec3> verts;
    for(int i=0;i<segs;i++){
        float a0 = float(i)/segs * glm::two_pi<float>();
        float a1 = float(i+1)/segs * glm::two_pi<float>();
        float r0 = innerR, r1 = outerR;
        float y0 = 0.02f * exp(-r0*4.5f);
        float y1 = 0.02f * exp(-r1*4.5f);
        verts.emplace_back(cos(a0)*r0, y0, sin(a0)*r0);
        verts.emplace_back(cos(a0)*r1, y1, sin(a0)*r1);
        verts.emplace_back(cos(a1)*r0, y0, sin(a1)*r0);
        verts.emplace_back(cos(a1)*r0, y0, sin(a1)*r0);
        verts.emplace_back(cos(a0)*r1, y1, sin(a0)*r1);
        verts.emplace_back(cos(a1)*r1, y1, sin(a1)*r1);
    }
    return verts;
}

// ------------------ Vertex / Fragment shaders (shared) ------------------

// Vertex shader: passa coordenadas locais do círculo (aPos.xy) ao fragment shader
const char* vertShared = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
out vec2 vLocal; // coordenadas locais do círculo, para blur
void main(){
    vLocal = aPos.xy; // para discos/billboards centrados na origem
    gl_Position = uMVP * vec4(aPos,1.0);
}
)";

// Fragment shaders:
// Grid simple white
const char* fragGrid = R"(
#version 330 core
out vec4 FragColor;
void main(){ FragColor = vec4(1.0,1.0,1.0,1.0); }
)";

// Black hole (com anel suave)
// Black hole (com anel suave)
const char* fragBlackHole = R"(
#version 330 core
in vec2 vLocal;
out vec4 FragColor;
void main(){
    float r = length(vLocal);
    // preto no centro
    if(r < 0.65) {
        FragColor = vec4(0.0,0.0,0.0,1.0);
        return;
    }
    // anel de fótons: suave laranja-vermelho
    float ring = smoothstep(0.65, 0.68, r) - smoothstep(0.72, 0.75, r);
    vec3 ringColor = vec3(1.0, 0.45, 0.1) * (0.6 + 0.8*ring);
    if(r < 0.85) {
        FragColor = vec4(ringColor, 1.0);
        return;
    }
    discard;
}
)";


// Disk (accreting) — mantém laranja/amarelo
const char* fragDisk = R"(
#version 330 core
in vec2 vLocal;
out vec4 FragColor;
void main(){
    float r = length(vLocal);
    // leve suavização radial
    float t = exp(-3.0*r);
    vec3 base = vec3(1.0,0.6,0.15);
    vec3 color = base * (0.6 + 0.8 * t);
    FragColor = vec4(color, 1.0);
}
)";

// Star fragment shader: usa vLocal para criar blur (gaussiano suave)
const char* fragStar = R"(
#version 330 core
in vec2 vLocal;
out vec4 FragColor;
uniform vec3 starColor;
uniform float blurIntensity; // controla força do blur
void main(){
    float r = length(vLocal);
    // gaussiana simples centrada, blurIntensity ajusta largura
    float sigma = 0.18 * blurIntensity; // menor sigma = mais pontual; maior = mais blur
    float intensity = exp(- (r*r) / (2.0 * sigma * sigma));
    intensity = clamp(intensity, 0.0, 1.0);
    vec3 col = starColor * (0.7 + 1.2*intensity);
    FragColor = vec4(col, intensity); // alpha = intensity para mistura suave
}
)";

// ------------------ Main program ------------------
int main(){
    // --- init GLFW / window
    if(!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800,600,"Black Hole Sim",nullptr,nullptr);
    if(!window){ glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK){ cerr<<"GLEW init failed\n"; return -1; }

    // register callbacks (same as before)
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // --- compile shaders (shared vert, multiple frags)
    GLuint gridShader = compileShaderSrc(vertShared, fragGrid);
    GLuint blackHoleShader = compileShaderSrc(vertShared, fragBlackHole);
    GLuint diskShader = compileShaderSrc(vertShared, fragDisk);
    GLuint starShader = compileShaderSrc(vertShared, fragStar);

    // --- generate geometry & upload buffers ---
    // grid
    generateGrid();

    // circle for black hole billboard (slightly larger)
    vector<vec3> circleVerts = generateCircle(0.9f, 96); // use radius ~0.9 (normalized for MVP scaling)
    GLuint circleVAO=0, circleVBO=0;
    glGenVertexArrays(1,&circleVAO);
    glGenBuffers(1,&circleVBO);
    glBindVertexArray(circleVAO);
    glBindBuffer(GL_ARRAY_BUFFER,circleVBO);
    glBufferData(GL_ARRAY_BUFFER,circleVerts.size()*sizeof(vec3),circleVerts.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    glBindVertexArray(0);

    // disk
    vector<vec3> diskVerts = generateDisk(0.35f, 0.85f, 100);
    GLuint diskVAO=0, diskVBO=0;
    glGenVertexArrays(1,&diskVAO);
    glGenBuffers(1,&diskVBO);
    glBindVertexArray(diskVAO);
    glBindBuffer(GL_ARRAY_BUFFER,diskVBO);
    glBufferData(GL_ARRAY_BUFFER,diskVerts.size()*sizeof(vec3),diskVerts.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    glBindVertexArray(0);

    // stars use same circle geometry (billboard) but with different size and color
    struct Star { vec3 pos; vec3 color; float size; float blur; GLuint VAO; GLuint VBO; };
    vector<Star> stars;

    // set stars: further away and larger, with blur
    stars.push_back({ vec3( 6.0f,  0.8f, -10.5f), vec3(1.0f, 0.12f, 0.08f), 1.2f, 1.4f, 0,0 }); // red distant (bigger)
    stars.push_back({ vec3(-7.0f,  0.6f, -11.2f), vec3(1.0f, 0.92f, 0.35f), 0.8f, 1.2f, 0,0 }); // yellow distant

    // prepare VAOs for stars (use same circleVerts)
    for(auto &s : stars){
        GLuint vao=0,vbo=0;
        glGenVertexArrays(1,&vao);
        glGenBuffers(1,&vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,circleVerts.size()*sizeof(vec3),circleVerts.data(),GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
        glBindVertexArray(0);
        s.VAO = vao; s.VBO = vbo;
    }

    // --- camera / projection
    mat4 proj = perspective(radians(60.0f), 800.0f/600.0f, 0.1f, 100.0f);

    // small tweak: lower black hole relative to grid. We'll place it near y = -0.35
    vec3 blackHolePos = vec3(0.0f, -0.35f, 0.0f);

    // main loop
    while(!glfwWindowShouldClose(window)){
        glClearColor(0.0f,0.0f,0.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mat4 view = lookAt(camera.position(), camera.target, vec3(0,1,0));
        mat4 VP = proj * view;

        // ---------------- Grid ----------------
        glUseProgram(gridShader);
        glUniformMatrix4fv(glGetUniformLocation(gridShader,"uMVP"),1,GL_FALSE,value_ptr(VP));
        glBindVertexArray(gridVAO);
        glDrawElements(GL_LINES, (GLsizei)gridIdx.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // ---------------- Disk ----------------
        // position disk slightly around blackHolePos, with scale so it visually sits on grid
        mat4 diskModel = translate(mat4(1.0f), blackHolePos + vec3(0.0f,-0.03f,0.0f));
        diskModel = scale(diskModel, vec3(1.0f));
        mat4 diskMVP = VP * diskModel;
        glUseProgram(diskShader);
        glUniformMatrix4fv(glGetUniformLocation(diskShader,"uMVP"),1,GL_FALSE,value_ptr(diskMVP));
        glBindVertexArray(diskVAO);
        // diskVerts are arranged as triangle lists
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)diskVerts.size());
        glBindVertexArray(0);

        // ---------------- Black hole (billboard) ----------------
        // billboard oriented to face camera; compute right/up/look for billboard basis
        vec3 objPos = blackHolePos;
        vec3 camPos = camera.position();
        vec3 look = normalize(camPos - objPos);
        vec3 right = normalize(cross(vec3(0,1,0), look));
        vec3 up = cross(look, right);

        mat4 model = mat4(1.0f);
        // normalize basis into mat3 columns; place in model matrix
        model[0] = vec4(right, 0.0f);
        model[1] = vec4(up, 0.0f);
        model[2] = vec4(look, 0.0f);
        model[3] = vec4(objPos, 1.0f);

        // scale so the black hole is a good visual size relative to grid
        float bhScale = 0.55f; // tweak: reduce or expand
        model = model * scale(mat4(1.0f), vec3(bhScale));

        mat4 mvp = VP * model;

        glUseProgram(blackHoleShader);
        glUniformMatrix4fv(glGetUniformLocation(blackHoleShader,"uMVP"),1,GL_FALSE,value_ptr(mvp));
        glBindVertexArray(circleVAO);
        glDrawArrays(GL_TRIANGLE_FAN, 0, (GLsizei)circleVerts.size());
        glBindVertexArray(0);


        // ---------------- Stars ----------------
        glUseProgram(starShader);
        for(auto &s : stars){
            // create an oriented billboard for each star (so it always faces camera)
            vec3 toCam = normalize(camPos - s.pos);
            vec3 rvec = normalize(cross(vec3(0,1,0), toCam));
            vec3 uvec = cross(toCam, rvec);

            mat4 starModel = mat4(1.0f);
            starModel[0] = vec4(rvec * s.size, 0.0f);
            starModel[1] = vec4(uvec * s.size, 0.0f);
            starModel[2] = vec4(toCam * s.size, 0.0f);
            starModel[3] = vec4(s.pos, 1.0f);

            mat4 starMVP = VP * starModel;
            glUniformMatrix4fv(glGetUniformLocation(starShader,"uMVP"),1,GL_FALSE,value_ptr(starMVP));
            glUniform3fv(glGetUniformLocation(starShader,"starColor"),1,value_ptr(s.color));
            glUniform1f(glGetUniformLocation(starShader,"blurIntensity"), s.blur);

            glBindVertexArray(s.VAO);
            glDrawArrays(GL_TRIANGLE_FAN, 0, (GLsizei)circleVerts.size());
            glBindVertexArray(0);
        }

        // swap / poll
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // cleanup (delete buffers)
    glDeleteVertexArrays(1,&circleVAO);
    glDeleteBuffers(1,&circleVBO);
    glDeleteVertexArrays(1,&diskVAO);
    glDeleteBuffers(1,&diskVBO);
    for(auto &s: stars){
        glDeleteVertexArrays(1, &s.VAO);
        glDeleteBuffers(1, &s.VBO);
    }
    glDeleteProgram(gridShader);
    glDeleteProgram(blackHoleShader);
    glDeleteProgram(diskShader);
    glDeleteProgram(starShader);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
