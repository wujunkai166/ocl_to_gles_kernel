#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl31.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

static const char gComputeShader[] = 
    "#version 310 es\n"
    "layout(local_size_x = 8, local_size_y = 8) in;\n"
    "layout(binding = 0) readonly buffer Input0 {\n"
    "    float data[];\n"
    "} input0;\n"
    "layout(binding = 1) readonly buffer Input1 {\n"
    "    int value;\n"
    "} input_w;\n"
    "layout(binding = 2) readonly buffer Input2 {\n"
    "    int value;\n"
    "} in_offset_x;\n"
    "layout(binding = 3) writeonly buffer Output {\n"
    "    float data[];\n"
    "} output0;\n"
    "layout(binding = 4) readonly buffer Input3 {\n"
    "    int value;\n"
    "} output_w;\n"
    "layout(binding = 5) readonly buffer Input4 {\n"
    "    int value;\n"
    "} out_offset_x;\n"
    "layout(binding = 6) readonly buffer Input5 {\n"
    "    int value;\n"
    "} max_g_x;\n"
    "layout(binding = 7) readonly buffer Input6 {\n"
    "    int value;\n"
    "} max_g_y;\n"
    "void main()\n"
    "{\n"
    "    int g_x = int(gl_GlobalInvocationID.x);\n"
    "    int g_y = int(gl_GlobalInvocationID.y);\n"
    "\n"
    "    if ((g_x >= max_g_x.value) || (g_y >= max_g_y.value))\n"
    "        return;\n"
    "\n"
    "    output0.data[g_y * output_w.value + g_x + out_offset_x.value] = input0.data[g_y * input_w.value + g_x + in_offset_x.value];\n"
    "}\n";

#define CHECK() \
{\
    GLenum err = glGetError(); \
    if (err != GL_NO_ERROR) \
    {\
        printf("glGetError returns %d\n", err); \
    }\
}

GLuint loadShader(GLenum shaderType, const char* pSource) {
    GLuint shader = glCreateShader(shaderType);
    if (shader) {
        glShaderSource(shader, 1, &pSource, NULL);
        glCompileShader(shader);
        GLint compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            GLint infoLen = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
            if (infoLen) {
                char* buf = (char*) malloc(infoLen);
                if (buf) {
                    glGetShaderInfoLog(shader, infoLen, NULL, buf);
                    fprintf(stderr, "Could not compile shader %d:\n%s\n",
                            shaderType, buf);
                    free(buf);
                }
                glDeleteShader(shader);
                shader = 0;
            }
        }
    }
    return shader;
}

GLuint createComputeProgram(const char* pComputeSource) {
    GLuint computeShader = loadShader(GL_COMPUTE_SHADER, pComputeSource);
    if (!computeShader) {
        return 0;
    }

    GLuint program = glCreateProgram();
    if (program) {
        glAttachShader(program, computeShader);
        glLinkProgram(program);
        GLint linkStatus = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
        if (linkStatus != GL_TRUE) {
            GLint bufLength = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
            if (bufLength) {
                char* buf = (char*) malloc(bufLength);
                if (buf) {
                    glGetProgramInfoLog(program, bufLength, NULL, buf);
                    fprintf(stderr, "Could not link program:\n%s\n", buf);
                    free(buf);
                }
            }
            glDeleteProgram(program);
            program = 0;
        }
    }
    return program;
}

void setupSSBufferObject(GLuint& ssbo, GLuint index, void* pIn, GLuint buf_size)
{
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);

    glBufferData(GL_SHADER_STORAGE_BUFFER, buf_size, pIn, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, ssbo);
}

void tryComputeShader()
{
    GLuint computeProgram;
    GLuint input0_ssbo;
    GLuint input1_ssbo;
    GLuint input2_ssbo;
    GLuint input3_ssbo;
    GLuint input4_ssbo;
    GLuint input5_ssbo;
    GLuint input6_ssbo;
    GLuint output0_ssbo;

    CHECK();
    computeProgram = createComputeProgram(gComputeShader);
    CHECK();

    cv::Mat input_image = cv::imread("input_1080p.jpg", CV_LOAD_IMAGE_COLOR);
    GLint image_w = input_image.cols;
    GLint image_h = input_image.rows;

    printf("w:%d, h:%d\n", image_w, image_h);

    GLuint image_size = image_w * image_h;
    float* input_bgr = (float *)malloc(image_size * 3 * sizeof(float));
    uint8_t* output_bgr = (uint8_t *)malloc(image_size * 3 * sizeof(uint8_t));

    for (GLuint i = 0; i < image_size * 3; i++) {
        input_bgr[i] = (float)input_image.data[i];
        output_bgr[i] = 0;
    }

    int in_offset_x = image_w * 3 / 4;
    int out_offset_x = 0;
    int max_g_x = image_w * 3 / 4;
    int max_g_y = image_h;

    int input_w = image_w * 3;
    int output_w = image_w * 3;

    setupSSBufferObject(input0_ssbo, 0, (void *)input_bgr, image_size * 3 * sizeof(float));
    setupSSBufferObject(input1_ssbo, 1, (void *)&input_w, sizeof(int));
    setupSSBufferObject(input2_ssbo, 2, (void *)&in_offset_x, sizeof(int));
    setupSSBufferObject(output0_ssbo, 3, NULL, image_size * 3 * sizeof(float));
    setupSSBufferObject(input3_ssbo, 4, (void *)&output_w, sizeof(int));
    setupSSBufferObject(input4_ssbo, 5, (void *)&out_offset_x, sizeof(int));
    setupSSBufferObject(input5_ssbo, 6, (void *)&max_g_x, sizeof(int));
    setupSSBufferObject(input6_ssbo, 7, (void *)&max_g_y, sizeof(int));

    CHECK();

    struct timeval start;
    struct timeval end;
    unsigned long diff = 0;
    gettimeofday(&start, NULL);

    float *pOut;
    for(int i = 0; i < 100; i++) {
        glUseProgram(computeProgram);
        glDispatchCompute(image_w * 3 / 8, image_h / 8, 1);   // arraySize/local_size_x
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        CHECK();

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output0_ssbo);
        pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size * sizeof(float), GL_MAP_READ_BIT);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    gettimeofday(&end, NULL);

    diff = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 100;
    printf("diff = %ld us\n", diff);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output0_ssbo);
    pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size * sizeof(float), GL_MAP_READ_BIT);
    for (GLuint i = 0; i < image_size * 3; i++) {
        output_bgr[i] = (uint8_t)pOut[i];
    }

    FILE *fp = fopen("output.raw", "w");
    fwrite(output_bgr, 1, image_size * 3, fp);
    fclose(fp);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glDeleteProgram(computeProgram);
}

int main(int /*argc*/, char** /*argv*/)
{
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) {
        printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
        return 0;
    }

    EGLint majorVersion;
    EGLint minorVersion;
    EGLBoolean returnValue = eglInitialize(dpy, &majorVersion, &minorVersion);
    if (returnValue != EGL_TRUE) {
        printf("eglInitialize failed\n");
        return 0;
    }

    EGLConfig cfg;
    EGLint count;
    EGLint s_configAttribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
            EGL_NONE };
    if (eglChooseConfig(dpy, s_configAttribs, &cfg, 1, &count) == EGL_FALSE) {
        printf("eglChooseConfig failed\n");
        return 0;
    }

    EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext context = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) {
        printf("eglCreateContext failed\n");
        return 0;
    }
    returnValue = eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
    if (returnValue != EGL_TRUE) {
        printf("eglMakeCurrent failed returned %d\n", returnValue);
        return 0;
    }

    tryComputeShader();

    eglDestroyContext(dpy, context);
    eglTerminate(dpy);

    return 0;
}
