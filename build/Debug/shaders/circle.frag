#include <GL/glew.h> 
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace glm;

// ================= Camera =================
struct Camera {
    vec3 target = vec3(0.0f);
    float radius = 2.5f;
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

// ================= Grid =================
vector<vec3> gridVerts;
vector<GLuint> gridIdx;
GLuint gridVAO = 0, gridVBO = 0, gridEBO = 0;

void generateGrid(int gridSize = 20, float spacing = 0.1f){
    gridVerts.clear();
    gridIdx.clear();
    for(int z = 0; z <= gridSize; ++z){
        for(int x = 0; x <= gridSize; ++x){
            float wx = (x - gridSize/2) * spacing;
            float wz = (z - gridSize/2) * spacing;
            float dist = sqrt(wx*wx + wz*wz);
            float y = -0.3f * exp(-dist*dist * 4.0f); // curvatura invertida
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
    if(!ok){ char buf[512]; glGetShaderInfoLog(v,512,nullptr,buf); cerr<<"Vert shader error: "<<buf<<endl;}

    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fragSrc, nullptr);
    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if(!ok){ char buf[512]; glGetShaderInfoLog(f,512,nullptr,buf); cerr<<"Frag shader error: "<<buf<<endl;}

    GLuint p = glCreateProgram();
    glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if(!ok){ char buf[512]; glGetProgramInfoLog(p,512,nullptr,buf); cerr<<"Program link error: "<<buf<<endl;}
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ================= Billboards =================
vector<vec3> generateCircle(float r=0.25f, int segs=64){
    vector<vec3> verts;
    verts.emplace_back(0,0,0);
    for(int i=0;i<=segs;i++){
        float a = float(i)/segs * glm::two_pi<float>();
        verts.emplace_back(cos(a)*r, sin(a)*r,0);
    }
    return verts;
}

// ================= Disco de acreção curvado =================
vector<vec3> generateDisk(float innerR=0.3f, float outerR=0.6f, int segs=128){
    vector<vec3> verts;
    for(int i=0;i<segs;i++){
        float a0 = float(i)/segs * glm::two_pi<float>();
        float a1 = float(i+1)/segs * glm::two_pi<float>();

        float r0 = innerR;
        float r1 = outerR;

        float y0 = 0.05f * exp(-r0*5.0f); // curvatura leve
        float y1 = 0.05f * exp(-r1*5.0f);

        // quad por segmento (2 triângulos)
        verts.emplace_back(cos(a0)*r0, y0, sin(a0)*r0);
        verts.emplace_back(cos(a0)*r1, y1, sin(a0)*r1);
        verts.emplace_back(cos(a1)*r0, y0, sin(a1)*r0);

        verts.emplace_back(cos(a1)*r0, y0, sin(a1)*r0);
        verts.emplace_back(cos(a0)*r1, y1, sin(a0)*r1);
        verts.emplace_back(cos(a1)*r1, y1, sin(a1)*r1);
    }
    return verts;
}

// ================= Main =================
int main(){
    if(!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(800,600,"Black Hole Sim",nullptr,nullptr);
    if(!window){ glfwTerminate(); return -1;}
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if(glewInit()!=GLEW_OK){ cerr<<"GLEW failed\n"; return -1;}

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glEnable(GL_DEPTH_TEST);

    // === Grid shader ===
    const char* gridVert = R"(
    #version 330 core
    layout(location=0) in vec3 aPos;
    uniform mat4 uMVP;
    void main(){ gl_Position = uMVP * vec4(aPos,1.0); })";

    const char* gridFrag = R"(
    #version 330 core
    out vec4 FragColor;
    void main(){ FragColor = vec4(1.0,1.0,1.0,1.0); })";

    GLuint gridShader = compileShaderSrc(gridVert, gridFrag);
    generateGrid();

    // === Circle black hole ===
    vector<vec3> circleVerts = generateCircle();
    GLuint circleVAO, circleVBO;
    glGenVertexArrays(1,&circleVAO);
    glGenBuffers(1,&circleVBO);
    glBindVertexArray(circleVAO);
    glBindBuffer(GL_ARRAY_BUFFER,circleVBO);
    glBufferData(GL_ARRAY_BUFFER,circleVerts.size()*sizeof(vec3),circleVerts.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    glBindVertexArray(0);

    // === Shader do círculo preto ===
    const char* circleFrag = R"(
    #version 330 core
    out vec4 FragColor;
    void main(){ FragColor = vec4(0.0,0.0,0.0,1.0); })";
    GLuint circleShader = compileShaderSrc(gridVert, circleFrag);

    // === Disk accretion ===
    vector<vec3> diskVerts = generateDisk();
    GLuint diskVAO, diskVBO;
    glGenVertexArrays(1,&diskVAO);
    glGenBuffers(1,&diskVBO);
    glBindVertexArray(diskVAO);
    glBindBuffer(GL_ARRAY_BUFFER,diskVBO);
    glBufferData(GL_ARRAY_BUFFER,diskVerts.size()*sizeof(vec3),diskVerts.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(vec3),(void*)0);
    glBindVertexArray(0);

    // === Disk shader ===
    const char* diskVert = R"(
    #version 330 core
    layout(location=0) in vec3 aPos;
    uniform mat4 uMVP;
    void main(){ gl_Position = uMVP * vec4(aPos,1.0); })";

    const char* diskFrag = R"(
    #version 330 core
    out vec4 FragColor;
    void main(){ FragColor = vec4(1.0,0.7,0.2,1.0); })";

    GLuint diskShader = compileShaderSrc(diskVert,diskFrag);

    while(!glfwWindowShouldClose(window)){
        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mat4 view = lookAt(camera.position(), camera.target, vec3(0,1,0));
        mat4 proj = perspective(radians(60.0f),800.0f/600.0f,0.1f,100.0f);
        mat4 VP = proj*view;

        // --- Grid ---
        glUseProgram(gridShader);
        glUniformMatrix4fv(glGetUniformLocation(gridShader,"uMVP"),1,GL_FALSE,value_ptr(VP));
        glBindVertexArray(gridVAO);
        glDrawElements(GL_LINES, gridIdx.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // --- Disk ---
        glUseProgram(diskShader);
        glUniformMatrix4fv(glGetUniformLocation(diskShader,"uMVP"),1,GL_FALSE,value_ptr(VP));
        glBindVertexArray(diskVAO);
        glDrawArrays(GL_TRIANGLES,0,diskVerts.size());
        glBindVertexArray(0);

        // --- Black hole circle (billboard completo) ---
        vec3 objPos = camera.target;
        vec3 camPos = camera.position();
        vec3 look = normalize(camPos - objPos);
        vec3 right = normalize(cross(vec3(0,1,0), look));
        vec3 up = cross(look, right);

        mat4 model = mat4(1.0f);
        model[0] = vec4(right, 0.0f);
        model[1] = vec4(up, 0.0f);
        model[2] = vec4(look, 0.0f);
        model[3] = vec4(objPos,1.0f);

        mat4 MVP = VP * model;

        glUseProgram(circleShader);
        glUniformMatrix4fv(glGetUniformLocation(circleShader,"uMVP"),1,GL_FALSE,value_ptr(MVP));
        glBindVertexArray(circleVAO);
        glDrawArrays(GL_TRIANGLE_FAN,0,circleVerts.size());
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}











