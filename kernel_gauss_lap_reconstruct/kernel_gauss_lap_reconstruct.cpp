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
    "layout(local_size_x = 8, local_size_y = 4) in;\n"
    "layout(binding = 0) readonly buffer Input0 {\n"
    "    vec4 data[];\n"
    "} input_lap;\n"
    "layout(binding = 1) readonly buffer Input1 {\n"
    "    int value;\n"
    "} input_lap_w;\n"
    "layout(binding = 2) readonly buffer Input2 {\n"
    "    vec2 data[];\n"
    "} input_gauss;\n"
    "layout(binding = 3) readonly buffer Input3 {\n"
    "    int value;\n"
    "} input_gauss_w;\n"
    "layout(binding = 4) readonly buffer Input4 {\n"
    "    int value;\n"
    "} input_gauss_h;\n"
    "layout(binding = 5) writeonly buffer Output {\n"
    "    vec4 data[];\n"
    "} output0;\n"
    "layout(binding = 6) readonly buffer Input5 {\n"
    "    int value;\n"
    "} output0_w;\n"
    "layout(binding = 7) readonly buffer Input6 {\n"
    "    int value;\n"
    "} out_offset_x;\n"
    "void main()\n"
    "{\n"
    "    int g_x = int(gl_GlobalInvocationID.x);\n"
    "    int cur_x = g_x;"
    "    int g_y = int(gl_GlobalInvocationID.y) * 2;\n"
    "\n"
    "    vec2 result_pre[2] = vec2[](vec2(0.0f), vec2(0.0f));\n"
    "    vec2 result_cur[2] = vec2[](vec2(0.0f), vec2(0.0f));\n"
    "    vec2 result_next[2] = vec2[](vec2(0.0f), vec2(0.0f));\n"
    "    vec4 final_g;\n"
    "    vec4 out_data;\n"
    "    vec2 tmp_data;\n"
    "    int i_ver;\n"
    "\n"
    "    float coeffs[6] = float[](0.0f, 0.289f, 0.5f, 0.422f, 0.5f, 0.289f);\n"
    "\n"
    "    for (i_ver = -1; i_ver <= 1; i_ver++) {\n"
    "        int cur_g_y = clamp(g_y / 2 + i_ver, 0, input_gauss_h.value - 1);\n"
    "        int pre_x = clamp(cur_x - 1, 0, input_gauss_w.value - 1);\n"
    "        int next_x = clamp(cur_x + 1, 0, input_gauss_w.value - 1);\n"
    "        float coeff0 = coeffs[(i_ver + 1) * 2 + 1];\n"
    "        float coeff1 = coeffs[(i_ver + 1) * 2];\n"
    "        tmp_data = input_gauss.data[cur_g_y * input_gauss_w.value + pre_x];\n"
    "        result_pre[0] += tmp_data * coeff0;\n"
    "        result_pre[1] += tmp_data * coeff1;\n"
    "        tmp_data = input_gauss.data[cur_g_y * input_gauss_w.value + cur_x];\n"
    "        result_cur[0] += tmp_data * coeff0;\n"
    "        result_cur[1] += tmp_data * coeff1;\n"
    "        tmp_data = input_gauss.data[cur_g_y * input_gauss_w.value + next_x];\n"
    "        result_next[0] += tmp_data * coeff0;\n"
    "        result_next[1] += tmp_data * coeff1;\n"
    "    }\n"
    "    int i_line;\n"
    "    vec3 tmp_result;\n"
    "    for (i_line = 0; i_line < 2; ++i_line) {\n"
    "        tmp_result = vec3(result_pre[i_line].y, result_cur[i_line]) * coeffs[1] +\n"
    "                     vec3(result_cur[i_line], result_next[i_line].x) * coeffs[3] +\n"
    "                     vec3(result_cur[i_line].y, result_next[i_line]) * coeffs[5];\n"
    "        final_g.xz = tmp_result.xy;\n"
    "        final_g.yw = tmp_result.xy * coeffs[2] + tmp_result.yz * coeffs[4];\n"
    "        final_g = clamp (final_g + 0.5f, 0.0f, 255.0f);\n"
    "        vec4 lap = input_lap.data[(g_y + i_line) * input_lap_w.value + g_x];\n"
    "        lap = (lap - 128.0f) * 2.0f;\n"
    "        out_data = final_g + lap + 0.5f;\n"
    "        output0.data[(g_y + i_line) * output0_w.value + g_x + out_offset_x.value] = out_data;\n"
    "    }\n"
    "}\n";

static const char gComputeShader_uv[] = 
    "#version 310 es\n"
    "layout(local_size_x = 8, local_size_y = 2) in;\n"
    "layout(binding = 0) readonly buffer Input0 {\n"
    "    vec4 data[];\n"
    "} input_lap;\n"
    "layout(binding = 1) readonly buffer Input1 {\n"
    "    int value;\n"
    "} input_lap_w;\n"
    "layout(binding = 2) readonly buffer Input2 {\n"
    "    vec2 data[];\n"
    "} input_gauss;\n"
    "layout(binding = 3) readonly buffer Input3 {\n"
    "    int value;\n"
    "} input_gauss_w;\n"
    "layout(binding = 4) readonly buffer Input4 {\n"
    "    int value;\n"
    "} input_gauss_h;\n"
    "layout(binding = 5) writeonly buffer Output {\n"
    "    vec4 data[];\n"
    "} output0;\n"
    "layout(binding = 6) readonly buffer Input5 {\n"
    "    int value;\n"
    "} output0_w;\n"
    "layout(binding = 7) readonly buffer Input6 {\n"
    "    int value;\n"
    "} out_offset_x;\n"
    "void main()\n"
    "{\n"
    "    int g_x = int(gl_GlobalInvocationID.x);\n"
    "    int cur_x = g_x;"
    "    int g_y = int(gl_GlobalInvocationID.y) * 2;\n"
    "\n"
    "    vec2 result_pre[2] = vec2[](vec2(0.0f), vec2(0.0f));\n"
    "    vec2 result_cur[2] = vec2[](vec2(0.0f), vec2(0.0f));\n"
    "    vec2 result_next1[2] = vec2[](vec2(0.0f), vec2(0.0f));\n"
    "    vec2 result_next2[2] = vec2[](vec2(0.0f), vec2(0.0f));\n"
    "    vec4 final_g;\n"
    "    vec4 out_data;\n"
    "    vec2 tmp_data;\n"
    "    int i_ver;\n"
    "\n"
    "    float coeffs[6] = float[](0.0f, 0.289f, 0.5f, 0.422f, 0.5f, 0.289f);\n"
    "\n"
    "    for (i_ver = -1; i_ver <= 1; i_ver++) {\n"
    "        int cur_g_y = clamp(g_y / 2 + i_ver, 0, input_gauss_h.value - 1);\n"
    "        int pre_x = clamp(cur_x - 1, 0, input_gauss_w.value - 1);\n"
    "        int next1_x = clamp(cur_x + 1, 0, input_gauss_w.value - 1);\n"
    "        int next2_x = clamp(cur_x + 2, 0, input_gauss_w.value - 1);\n"
    "        float coeff0 = coeffs[(i_ver + 1) * 2 + 1];\n"
    "        float coeff1 = coeffs[(i_ver + 1) * 2];\n"
    "        tmp_data = input_gauss.data[cur_g_y * input_gauss_w.value + pre_x];\n"
    "        result_pre[0] += tmp_data * coeff0;\n"
    "        result_pre[1] += tmp_data * coeff1;\n"
    "        tmp_data = input_gauss.data[cur_g_y * input_gauss_w.value + cur_x];\n"
    "        result_cur[0] += tmp_data * coeff0;\n"
    "        result_cur[1] += tmp_data * coeff1;\n"
    "        tmp_data = input_gauss.data[cur_g_y * input_gauss_w.value + next1_x];\n"
    "        result_next1[0] += tmp_data * coeff0;\n"
    "        result_next1[1] += tmp_data * coeff1;\n"
    "        tmp_data = input_gauss.data[cur_g_y * input_gauss_w.value + next2_x];\n"
    "        result_next2[0] += tmp_data * coeff0;\n"
    "        result_next2[1] += tmp_data * coeff1;\n"
    "    }\n"
    "    int i_line;\n"
    "    vec4 tmp_result;\n"
    "    for (i_line = 0; i_line < 2; ++i_line) {\n"
    "        tmp_result = vec4(result_pre[i_line], result_cur[i_line]) * coeffs[1] +\n"
    "                     vec4(result_cur[i_line], result_next1[i_line]) * coeffs[3] +\n"
    "                     vec4(result_next1[i_line], result_next2[i_line]) * coeffs[5];\n"
    "        final_g.xy = tmp_result.xy;\n"
    "        final_g.zw = tmp_result.xy * coeffs[2] + tmp_result.zw * coeffs[4];\n"
    "        final_g = clamp (final_g + 0.5f, 0.0f, 255.0f);\n"
    "        vec4 lap = input_lap.data[(g_y + i_line) * input_lap_w.value + g_x];\n"
    "        lap = (lap - 128.0f) * 2.0f;\n"
    "        out_data = final_g + lap + 0.5f;\n"
    "        output0.data[(g_y + i_line) * output0_w.value + g_x + out_offset_x.value] = out_data;\n"
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
    GLuint input7_ssbo;
    GLuint output0_ssbo;

    CHECK();
    computeProgram = createComputeProgram(gComputeShader_y);
    CHECK();

    cv::Mat input0_image = cv::imread("input_1080p.jpg", CV_LOAD_IMAGE_COLOR);
    GLint image_w = input0_image.cols;
    GLint image_h = input0_image.rows;

    printf("w:%d, h:%d\n", image_w, image_h);

    GLuint image_size = image_w * image_h;

    uint8_t* input_gauss_nv12 = (uint8_t *)malloc(image_size * 3 / 8 * sizeof(uint8_t));
    float* input_gauss_y = (float *)malloc(image_size / 4 * sizeof(float));
    float* input_gauss_uv = (float *)malloc(image_size / 8 * sizeof(float));

    uint8_t* input_lap_nv12 = (uint8_t *)malloc(image_size * 3 / 2 * sizeof(uint8_t));
    float* input_lap_y = (float *)malloc(image_size * sizeof(float));
    float* input_lap_uv = (float *)malloc(image_size / 2 * sizeof(float));

    FILE* fp = fopen("input_gauss.raw", "r");
    fread(input_gauss_nv12, 1, image_size * 3 / 8, fp);
    fclose(fp);

    fp = fopen("input_lap.raw", "r");
    fread(input_lap_nv12, 1, image_size * 3 / 2, fp);
    fclose(fp);

    uint8_t* output_nv12 = (uint8_t *)malloc(image_size * 3 / 2 * sizeof(uint8_t));

    for (GLuint i = 0; i < image_size / 4; i++) {
        input_gauss_y[i] = (float)input_gauss_nv12[i];
    }
    
    for (GLuint i = 0; i < image_size / 8; i++) {
        input_gauss_uv[i] = (float)input_gauss_nv12[i + image_size / 4];
    }

    for (GLuint i = 0; i < image_size; i++) {
        input_lap_y[i] = (float)input_lap_nv12[i];
    }

    for (GLuint i = 0; i < image_size / 2; i++) {
        input_lap_uv[i] = (float)input_lap_nv12[i + image_size];
    }

    int input_w = image_w / 4;
    int input_h = image_h / 2;
    int output_w = image_w / 4;
    int out_offset_x = 0;

    setupSSBufferObject(input0_ssbo, 0, (void *)input_lap_y, image_size * sizeof(float));
    setupSSBufferObject(input1_ssbo, 1, (void *)&input_w, sizeof(int));
    setupSSBufferObject(input2_ssbo, 2, (void *)input_gauss_y, image_size / 4 * sizeof(float));
    setupSSBufferObject(input3_ssbo, 3, (void *)&input_w, sizeof(int));
    setupSSBufferObject(input4_ssbo, 4, (void *)&input_h, sizeof(int));
    setupSSBufferObject(output0_ssbo, 5, NULL, image_size * sizeof(float));
    setupSSBufferObject(input5_ssbo, 6, (void *)&output_w, sizeof(int));
    setupSSBufferObject(input6_ssbo, 7, (void *)&out_offset_x, sizeof(int));

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
        pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size * sizeof(float), GL_MAP_READ_BIT);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    gettimeofday(&end, NULL);

    diff = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 100;
    printf("diff = %ld us\n", diff);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output0_ssbo);
    pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size * sizeof(float), GL_MAP_READ_BIT);
    for (GLuint i = 0; i < image_size; i++) {
        output_nv12[i] = (uint8_t)pOut[i];
    }

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glDeleteProgram(computeProgram);

    computeProgram = createComputeProgram(gComputeShader_uv);
    CHECK();

    input_h = image_h / 4;

    setupSSBufferObject(input0_ssbo, 0, (void *)input_lap_uv, image_size / 2 * sizeof(float));
    setupSSBufferObject(input1_ssbo, 1, (void *)&input_w, sizeof(int));
    setupSSBufferObject(input3_ssbo, 2, (void *)input_gauss_uv, image_size / 8 * sizeof(float));
    setupSSBufferObject(input4_ssbo, 3, (void *)&input_w, sizeof(int));
    setupSSBufferObject(input5_ssbo, 4, (void *)&input_h, sizeof(int));
    setupSSBufferObject(output0_ssbo, 5, NULL, image_size / 2 * sizeof(float));
    setupSSBufferObject(input6_ssbo, 6, (void *)&output_w, sizeof(int));
    setupSSBufferObject(input7_ssbo, 7, (void *)&out_offset_x, sizeof(int));

    CHECK();

    gettimeofday(&start, NULL);

    for(int i = 0; i < 100; i++) {
        glUseProgram(computeProgram);
        glDispatchCompute(image_w / 32, image_h / 8, 1);   // arraySize/local_size_x
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        CHECK();

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output0_ssbo);
        pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size / 2 * sizeof(float), GL_MAP_READ_BIT);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    gettimeofday(&end, NULL);

    diff = (1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 100;
    printf("diff = %ld us\n", diff);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output0_ssbo);
    pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size / 2 * sizeof(float), GL_MAP_READ_BIT);
    for (GLuint i = 0; i < image_size / 2; i++) {
        output_nv12[i + image_size] = (uint8_t)pOut[i];
    }

    fp = fopen("output.raw", "w");
    fwrite(output_nv12, 1, image_size * 3 / 2, fp);
    fclose(fp);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glDeleteProgram(computeProgram);
    
    free(input_lap_nv12);
    free(input_lap_y);
    free(input_lap_uv);
    free(output_nv12);
    free(input_gauss_nv12);
    free(input_gauss_y);
    free(input_gauss_uv);
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
