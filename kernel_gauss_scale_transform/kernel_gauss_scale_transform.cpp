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

static const char gComputeShader_y[] = 
    "#version 310 es\n"
    "layout(local_size_x = 8, local_size_y = 2) in;\n"
    "layout(binding = 0) readonly buffer Input0 {\n"
    "    vec4 data[];\n"
    "} input0;\n"
    "layout(binding = 1) readonly buffer Input1 {\n"
    "    int value;\n"
    "} input0_w;\n"
    "layout(binding = 2) readonly buffer Input2 {\n"
    "    int value;\n"
    "} input0_h;\n"
    "layout(binding = 3) readonly buffer Input3 {\n"
    "    int value;\n"
    "} in_offset_x;\n"
    "layout(binding = 4) writeonly buffer Output {\n"
    "    vec2 data[];\n"
    "} output0;\n"
    "layout(binding = 5) readonly buffer Input4 {\n"
    "    int value;\n"
    "} output0_w;\n"
    "void main()\n"
    "{\n"
    "    int g_x = int(gl_GlobalInvocationID.x);\n"
    "    int cur_x = g_x + in_offset_x.value;\n"
    "    int g_y = int(gl_GlobalInvocationID.y) * 4;\n"
    "    int g_out_x = int(gl_GlobalInvocationID.x);\n"
    "    int g_out_y = int(gl_GlobalInvocationID.y) * 2;\n"
    "\n"
    "    vec4 result_pre[2] = vec4[](vec4(0.0f), vec4(0.0f));\n"
    "    vec4 result_cur[2] = vec4[](vec4(0.0f), vec4(0.0f));\n"
    "    vec4 result_next[2] = vec4[](vec4(0.0f), vec4(0.0f));\n"
    "    vec2 final_g[2];\n"
    "    vec4 tmp_data;\n"
    "    int i_ver;\n"
    "\n"
    "    float coeffs[9] = float[](0.0f, 0.0f, 0.152f, 0.222f, 0.252f, 0.222f, 0.152f, 0.0f, 0.0f);\n"
    "\n"
    "    for (i_ver = -2; i_ver <= 4; i_ver++) {\n"
    "        int cur_g_y = clamp(g_y + i_ver, 0, input0_h.value - 1);\n"
    "        int pre_x = clamp(cur_x - 1, 0, input0_w.value - 1);\n"
    "        int next_x = clamp(cur_x + 1, 0, input0_w.value - 1);\n"
    "        float coeff0 = coeffs[i_ver + 4];\n"
    "        float coeff1 = coeffs[i_ver + 2];\n"
    "        tmp_data = input0.data[cur_g_y * input0_w.value + pre_x];\n"
    "        result_pre[0] += tmp_data * coeff0;\n"
    "        result_pre[1] += tmp_data * coeff1;\n"
    "        tmp_data = input0.data[cur_g_y * input0_w.value + cur_x];\n"
    "        result_cur[0] += tmp_data * coeff0;\n"
    "        result_cur[1] += tmp_data * coeff1;\n"
    "        tmp_data = input0.data[cur_g_y * input0_w.value + next_x];\n"
    "        result_next[0] += tmp_data * coeff0;\n"
    "        result_next[1] += tmp_data * coeff1;\n"
    "    }\n"
    "    int i_line;\n"
    "    for (i_line = 0; i_line < 2; ++i_line) {\n"
    "        final_g[i_line] = vec2(result_cur[i_line].xz) * coeffs[4] +\n"
    "                          vec2(result_pre[i_line].w, result_cur[i_line].y) * coeffs[5] +\n"
    "                          vec2(result_pre[i_line].z, result_cur[i_line].x) * coeffs[6] +\n"
    "                          vec2(result_cur[i_line].yw) * coeffs[5] +\n"
    "                          vec2(result_cur[i_line].z, result_next[i_line].x) * coeffs[6];\n"
    "        final_g[i_line] = clamp (final_g[i_line] + 0.5f, 0.0f, 255.0f);\n"
    "        output0.data[(g_out_y + i_line) * output0_w.value + g_out_x] = final_g[i_line];\n"
    "    }\n"
    "}\n";

static const char gComputeShader_uv[] =
    "#version 310 es\n"
    "layout(local_size_x = 8, local_size_y = 1) in;\n"
    "layout(binding = 0) readonly buffer Input0 {\n"
    "    vec4 data[];\n"
    "} input0;\n"
    "layout(binding = 1) readonly buffer Input1 {\n"
    "    int value;\n"
    "} input0_w;\n"
    "layout(binding = 2) readonly buffer Input2 {\n"
    "    int value;\n"
    "} input0_h;\n"
    "layout(binding = 3) readonly buffer Input3 {\n"
    "    int value;\n"
    "} in_offset_x;\n"
    "layout(binding = 4) writeonly buffer Output {\n"
    "    vec2 data[];\n"
    "} output0;\n"
    "layout(binding = 5) readonly buffer Input4 {\n"
    "    int value;\n"
    "} output0_w;\n"
    "void main()\n"
    "{\n"
    "    int g_x = int(gl_GlobalInvocationID.x);\n"
    "    int cur_x = g_x + in_offset_x.value;\n"
    "    int g_y = int(gl_GlobalInvocationID.y) * 4;\n"
    "    int g_out_x = int(gl_GlobalInvocationID.x);\n"
    "    int g_out_y = int(gl_GlobalInvocationID.y) * 2;\n"
    "\n"
    "    vec4 result_pre[2] = vec4[](vec4(0.0f), vec4(0.0f));\n"
    "    vec4 result_cur[2] = vec4[](vec4(0.0f), vec4(0.0f));\n"
    "    vec4 result_next[2] = vec4[](vec4(0.0f), vec4(0.0f));\n"
    "    vec2 final_g[2];\n"
    "    vec4 tmp_data;\n"
    "    int i_ver;\n"
    "\n"
    "    float coeffs[9] = float[](0.0f, 0.0f, 0.152f, 0.222f, 0.252f, 0.222f, 0.152f, 0.0f, 0.0f);\n"
    "\n"
    "    for (i_ver = -2; i_ver <= 4; i_ver++) {\n"
    "        int cur_g_y = clamp(g_y + i_ver, 0, input0_h.value - 1);\n"
    "        int pre_x = clamp(cur_x - 1, 0, input0_w.value - 1);\n"
    "        int next_x = clamp(cur_x + 1, 0, input0_w.value - 1);\n"
    "        float coeff0 = coeffs[i_ver + 4];\n"
    "        float coeff1 = coeffs[i_ver + 2];\n"
    "        tmp_data = input0.data[cur_g_y * input0_w.value + pre_x];\n"
    "        result_pre[0] += tmp_data * coeff0;\n"
    "        result_pre[1] += tmp_data * coeff1;\n"
    "        tmp_data = input0.data[cur_g_y * input0_w.value + cur_x];\n"
    "        result_cur[0] += tmp_data * coeff0;\n"
    "        result_cur[1] += tmp_data * coeff1;\n"
    "        tmp_data = input0.data[cur_g_y * input0_w.value + next_x];\n"
    "        result_next[0] += tmp_data * coeff0;\n"
    "        result_next[1] += tmp_data * coeff1;\n"
    "    }\n"
    "    int i_line;\n"
    "    for (i_line = 0; i_line < 2; ++i_line) {\n"
    "        final_g[i_line] = vec2(result_cur[i_line].xy) * coeffs[4] +\n"
    "                          vec2(result_pre[i_line].zw) * coeffs[5] +\n"
    "                          vec2(result_pre[i_line].xy) * coeffs[6] +\n"
    "                          vec2(result_cur[i_line].zw) * coeffs[5] +\n"
    "                          vec2(result_next[i_line].xy) * coeffs[6];\n"
    "        final_g[i_line] = clamp (final_g[i_line] + 0.5f, 0.0f, 255.0f);\n"
    "        output0.data[(g_out_y + i_line) * output0_w.value + g_out_x] = final_g[i_line];\n"
    "    }\n"
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

void convert_bgr_to_nv12 (float* input, float* output_y, float* output_uv, int width, int height)
{
    float *y = (float *) malloc (width * height * sizeof (float));
    float *u = (float *) malloc (width * height * sizeof (float));
    float *v = (float *) malloc (width * height * sizeof (float));

    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            int index = row * width + col;
            float b = input[index * 3];
            float g = input[index * 3 + 1];
            float r = input[index * 3 + 2];

            y[index] = 0.299f * r + 0.587f * g + 0.114f * b;
            u[index] = -0.169f * r - 0.331f * g + 0.5f * b + 128;
            v[index] = 0.5f * r - 0.419f * g - 0.081f * b + 128;

            output_y[index] = y[index];
        }
    }

    for(int row = 0; row < height / 2; row++) {
        for(int col = 0; col < width / 2; col++) {
            int index = row * width / 2 + col;

            output_uv[index * 2] = u[row * 2 * width + col * 2];
            output_uv[index * 2 + 1] = v[row * 2 * width + col * 2];
        }
    }

    free(y);
    free(u);
    free(v);
}

void tryComputeShader()
{
    GLuint computeProgram;
    GLuint input0_ssbo;
    GLuint input1_ssbo;
    GLuint input2_ssbo;
    GLuint input3_ssbo;
    GLuint input4_ssbo;
    GLuint output0_ssbo;

    CHECK();
    computeProgram = createComputeProgram(gComputeShader_y);
    CHECK();

    cv::Mat input0_image = cv::imread("input_1080p.jpg", CV_LOAD_IMAGE_COLOR);
    GLint image_w = input0_image.cols;
    GLint image_h = input0_image.rows;

    printf("w:%d, h:%d\n", image_w, image_h);

    GLuint image_size = image_w * image_h;
    float* input0_bgr = (float *)malloc(image_size * 3 * sizeof(float));
    float* input0_y = (float *)malloc(image_size * sizeof(float));
    float* input0_uv = (float *)malloc(image_size / 2 * sizeof(float));

    uint8_t* output_nv12 = (uint8_t *)malloc(image_size * 3 / 8 * sizeof(uint8_t));

    for (GLuint i = 0; i < image_size * 3 / 8; i++) {
        output_nv12[i] = 1;
    }

    for (GLuint i = 0; i < image_size * 3; i++) {
        input0_bgr[i] = (float)input0_image.data[i];
    }

    convert_bgr_to_nv12 (input0_bgr, input0_y, input0_uv, image_w, image_h);

    int input_w = image_w / 4;
    int output_w = image_w / 4;
    int in_offset_x = 0;

    setupSSBufferObject(input0_ssbo, 0, (void *)input0_y, image_size * sizeof(float));
    setupSSBufferObject(input1_ssbo, 1, (void *)&input_w, sizeof(int));
    setupSSBufferObject(input2_ssbo, 2, (void *)&image_h, sizeof(int));
    setupSSBufferObject(input3_ssbo, 3, (void *)&in_offset_x, sizeof(int));
    setupSSBufferObject(output0_ssbo, 4, NULL, image_size / 4 * sizeof(float));
    setupSSBufferObject(input4_ssbo, 5, (void *)&output_w, sizeof(int));

    CHECK();

    struct timeval start;
    struct timeval end;
    unsigned long diff = 0;
    gettimeofday(&start, NULL);

    float *pOut;
    for(int i = 0; i < 100; i++) {
        glUseProgram(computeProgram);
        glDispatchCompute(image_w / 32, image_h / 8, 1);   // arraySize/local_size_x
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        CHECK();

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output0_ssbo);
        pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size / 4 * sizeof(float), GL_MAP_READ_BIT);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    gettimeofday(&end, NULL);

    diff = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 100;
    printf("diff = %ld us\n", diff);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output0_ssbo);
    pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size / 4 * sizeof(float), GL_MAP_READ_BIT);
    for (GLuint i = 0; i < image_size / 4; i++) {
        output_nv12[i] = (uint8_t)pOut[i];
    }

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glDeleteProgram(computeProgram);

    computeProgram = createComputeProgram(gComputeShader_uv);
    CHECK();

    int input_h = image_h / 2;

    setupSSBufferObject(input0_ssbo, 0, (void *)input0_uv, image_size / 2 * sizeof(float));
    setupSSBufferObject(input1_ssbo, 1, (void *)&input_w, sizeof(int));
    setupSSBufferObject(input2_ssbo, 2, (void *)&input_h, sizeof(int));
    setupSSBufferObject(input3_ssbo, 3, (void *)&in_offset_x, sizeof(int));
    setupSSBufferObject(output0_ssbo, 4, NULL, image_size / 8 * sizeof(float));
    setupSSBufferObject(input4_ssbo, 5, (void *)&output_w, sizeof(int));

    CHECK();

    gettimeofday(&start, NULL);

    for(int i = 0; i < 100; i++) {
        glUseProgram(computeProgram);
        glDispatchCompute(image_w / 32, image_h / 8, 1);   // arraySize/local_size_x
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        CHECK();

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output0_ssbo);
        pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size / 8 * sizeof(float), GL_MAP_READ_BIT);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    gettimeofday(&end, NULL);

    diff = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 100;
    printf("diff = %ld us\n", diff);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output0_ssbo);
    pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size / 8 * sizeof(float), GL_MAP_READ_BIT);
    for (GLuint i = 0; i < image_size / 8; i++) {
        output_nv12[i + image_size / 4] = (uint8_t)pOut[i];
    }

    FILE *fp = fopen("output.raw", "w");
    fwrite(output_nv12, 1, image_size * 3 / 8, fp);
    fclose(fp);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glDeleteProgram(computeProgram);

    free(input0_bgr);
    free(input0_y);
    free(input0_uv);
    free(output_nv12);
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
