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
    "} input_y;\n"
    "layout(binding = 1) readonly buffer Input1 {\n"
    "    vec2 data[];\n"
    "} input_uv;\n"
    "layout(binding = 2) readonly buffer Input2 {\n"
    "    vec2 data[];\n"
    "} geo_table;\n"
    "layout(binding = 3) readonly buffer Input3 {\n"
    "    int data[];\n"
    "} table_scale_size;\n"
    "layout(binding = 4) readonly buffer Input4 {\n"
    "    int value;\n"
    "} image_width;\n"
    "layout(binding = 5) readonly buffer Input5 {\n"
    "    int value;\n"
    "} image_height;\n"
    "layout(binding = 6) writeonly buffer Output0 {\n"
    "    vec2 data[];\n"
    "} output_y;\n"
    "layout(binding = 7) writeonly buffer Output1 {\n"
    "    vec2 data[];\n"
    "} output_uv;\n"
    "void interpolation(in vec2 input_pos, out ivec2 output_pos[4], out vec4 output_weight)\n"
    "{\n"
    "    int x0 = int(input_pos.x);\n"
    "    int y0 = int(input_pos.y);\n"
    "    int x1 = x0 + 1;\n"
    "    int y1 = y0 + 1;\n"
    "\n"
    "    output_pos[0] = ivec2(x0, y0);\n"
    "    output_pos[1] = ivec2(x1, y0);\n"
    "    output_pos[2] = ivec2(x0, y1);\n"
    "    output_pos[3] = ivec2(x1, y1);\n"
    "\n"
    "    float diff_x = input_pos.x - float(x0);\n"
    "    float diff_y = input_pos.y - float(y0);\n"
    "\n"
    "    output_weight.x = (1.0f - diff_x) * (1.0f - diff_y);"
    "    output_weight.y = diff_x * (1.0f - diff_y);"
    "    output_weight.z = (1.0f - diff_x) * diff_y;"
    "    output_weight.w = diff_x * diff_y;"
    "}\n"
    "void main()\n"
    "{\n"
    "    int col = int(gl_GlobalInvocationID.x) * 2;\n"
    "    int row = int(gl_GlobalInvocationID.y) * 2;\n"
    "\n"
    "    vec2 out_map_pos;\n"
    "    vec2 input_pos[2];\n"
    "    bool out_of_bound[2];"
    "    vec2 table_scale_step = 1.0f / vec2(float(table_scale_size.data[0]), float(table_scale_size.data[1]));\n"
    "    int geo_table_width = int(float(image_width.value) * table_scale_step.x);\n"
    "    out_map_pos = vec2(float(col), float(row)) * table_scale_step;\n"
    "\n"
    "    ivec2 inter_pos[4];\n"
    "    vec4 inter_weight;\n"
    "    vec4 map_index_x;\n"
    "    vec4 map_index_y;\n"
    "    vec4 map_data;\n"
    "    vec2 output_value;\n"
    "\n"
    "    for(int i = 0; i < 2; i++) {\n"
    "        interpolation(out_map_pos, inter_pos, inter_weight);\n"
    "\n"
    "        vec2 geo_table_value[4];\n"
    "        geo_table_value[0] = geo_table.data[inter_pos[0].y * geo_table_width + inter_pos[0].x];\n"
    "        geo_table_value[1] = geo_table.data[inter_pos[1].y * geo_table_width + inter_pos[1].x];\n"
    "        geo_table_value[2] = geo_table.data[inter_pos[2].y * geo_table_width + inter_pos[2].x];\n"
    "        geo_table_value[3] = geo_table.data[inter_pos[3].y * geo_table_width + inter_pos[3].x];\n"
    "        map_index_x.x = geo_table_value[0].x;\n"
    "        map_index_y.x = geo_table_value[0].y;\n"
    "        map_index_x.y = geo_table_value[1].x;\n"
    "        map_index_y.y = geo_table_value[1].y;\n"
    "        map_index_x.z = geo_table_value[2].x;\n"
    "        map_index_y.z = geo_table_value[2].y;\n"
    "        map_index_x.w = geo_table_value[3].x;\n"
    "        map_index_y.w = geo_table_value[3].y;\n"
    "\n"
    "        input_pos[i].x = dot(map_index_x, inter_weight);\n"
    "        input_pos[i].y = dot(map_index_y, inter_weight);\n"
    "\n"
    "        out_of_bound[i] = (min(input_pos[i].x, input_pos[i].y) < 0.0f) || (input_pos[i].x > float(image_width.value)) || (input_pos[i].y > float(image_height.value));\n"
    "        if(!out_of_bound[i]) {\n"
    "            interpolation(input_pos[i], inter_pos, inter_weight);\n"
    "            map_data.x = input_y.data[inter_pos[0].y * image_width.value + inter_pos[0].x];\n"
    "            map_data.y = input_y.data[inter_pos[1].y * image_width.value + inter_pos[1].x];\n"
    "            map_data.z = input_y.data[inter_pos[2].y * image_width.value + inter_pos[2].x];\n"
    "            map_data.w = input_y.data[inter_pos[3].y * image_width.value + inter_pos[3].x];\n"
    "            output_value[i] = dot(map_data, inter_weight);\n"
    "        } else {\n"
    "            output_value[i] = 0.0f;\n"
    "        }\n"
    "\n"
    "        out_map_pos.x += table_scale_step.x;\n"
    "    }\n"
    "    output_y.data[(row * image_width.value + col) / 2] = output_value;"
    "\n"
    "    if(!out_of_bound[0]) {\n"
    "        input_pos[0] *= 0.5f;\n"
    "        interpolation(input_pos[0], inter_pos, inter_weight);\n"
    "        vec2 uv_value[4];\n"
    "        uv_value[0] = input_uv.data[inter_pos[0].y * image_width.value / 2 + inter_pos[0].x];\n"
    "        uv_value[1] = input_uv.data[inter_pos[1].y * image_width.value / 2 + inter_pos[1].x];\n"
    "        uv_value[2] = input_uv.data[inter_pos[2].y * image_width.value / 2 + inter_pos[2].x];\n"
    "        uv_value[3] = input_uv.data[inter_pos[3].y * image_width.value / 2 + inter_pos[3].x];\n"
    "        map_data.x = uv_value[0].x;\n"
    "        map_data.y = uv_value[1].x;\n"
    "        map_data.z = uv_value[2].x;\n"
    "        map_data.w = uv_value[3].x;\n"
    "        output_value.x = dot(map_data, inter_weight);\n"
    "        map_data.x = uv_value[0].y;\n"
    "        map_data.y = uv_value[1].y;\n"
    "        map_data.z = uv_value[2].y;\n"
    "        map_data.w = uv_value[3].y;\n"
    "        output_value.y = dot(map_data, inter_weight);\n"
    "        output_uv.data[(row / 2 * image_width.value + col) / 2] = output_value;\n"
    "    } else {\n"
    "        output_uv.data[(row / 2 * image_width.value + col) / 2] = vec2(0.5f, 0.5f);\n"
    "    }\n"
    "\n"
    "    out_map_pos.x -= 2.0f * table_scale_step.x;\n"
    "    out_map_pos.y += table_scale_step.y;\n"
    "    for(int i = 0; i < 2; i++) {\n"
    "        interpolation(out_map_pos, inter_pos, inter_weight);\n"
    "\n"
    "        vec2 geo_table_value[4];\n"
    "        geo_table_value[0] = geo_table.data[inter_pos[0].y * geo_table_width + inter_pos[0].x];\n"
    "        geo_table_value[1] = geo_table.data[inter_pos[1].y * geo_table_width + inter_pos[1].x];\n"
    "        geo_table_value[2] = geo_table.data[inter_pos[2].y * geo_table_width + inter_pos[2].x];\n"
    "        geo_table_value[3] = geo_table.data[inter_pos[3].y * geo_table_width + inter_pos[3].x];\n"
    "        map_index_x.x = geo_table_value[0].x;\n"
    "        map_index_y.x = geo_table_value[0].y;\n"
    "        map_index_x.y = geo_table_value[1].x;\n"
    "        map_index_y.y = geo_table_value[1].y;\n"
    "        map_index_x.z = geo_table_value[2].x;\n"
    "        map_index_y.z = geo_table_value[2].y;\n"
    "        map_index_x.w = geo_table_value[3].x;\n"
    "        map_index_y.w = geo_table_value[3].y;\n"
    "\n"
    "        input_pos[i].x = dot(map_index_x, inter_weight);\n"
    "        input_pos[i].y = dot(map_index_y, inter_weight);\n"
    "\n"
    "        out_of_bound[i] = (min(input_pos[i].x, input_pos[i].y) < 0.0f) || (input_pos[i].x > float(image_width.value)) || (input_pos[i].y > float(image_height.value));\n"
    "        if(!out_of_bound[i]) {\n"
    "            interpolation(input_pos[i], inter_pos, inter_weight);\n"
    "            map_data.x = input_y.data[inter_pos[0].y * image_width.value + inter_pos[0].x];\n"
    "            map_data.y = input_y.data[inter_pos[1].y * image_width.value + inter_pos[1].x];\n"
    "            map_data.z = input_y.data[inter_pos[2].y * image_width.value + inter_pos[2].x];\n"
    "            map_data.w = input_y.data[inter_pos[3].y * image_width.value + inter_pos[3].x];\n"
    "            output_value[i] = dot(map_data, inter_weight);\n"
    "        } else {\n"
    "            output_value[i] = 0.0f;\n"
    "        }\n"
    "\n"
    "        out_map_pos.x += table_scale_step.x;\n"
    "    }\n"
    "    output_y.data[((row + 1) * image_width.value + col) / 2] = output_value;"
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

void convert_nv12_to_bgr (float* input, float* output, int width, int height)
{
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            int index_y = row * width + col;
            int index_u = width * height + row / 2 * width + (col / 2 ) * 2;
            int index_v = width * height + row / 2 * width + (col / 2 ) * 2 + 1; 

            float y = input[index_y];
            float u = input[index_u];
            float v = input[index_v];

            input[index_y * 3] = y + 1.77f * (u - 128);
            input[index_y * 3 + 1] = y - 0.343f * (u - 128) - 0.714f * (v - 128);
            input[index_y * 3 + 2] = y + 1.403f * (v - 128);
        }
    }
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
    GLuint output0_ssbo;
    GLuint output1_ssbo;
   
    cv::Mat input_image = cv::imread("input_1080p.jpg", CV_LOAD_IMAGE_COLOR);
    cv::Mat output_image (input_image);
    GLint image_w = input_image.cols;
    GLint image_h = input_image.rows;

    printf("w:%d, h:%d\n", image_w, image_h);

    GLuint image_size = image_w * image_h;
    float* input_bgr = (float *)malloc(image_size * 3 * sizeof(float));
    float* input_y = (float *)malloc(image_size * sizeof(float));
    float* input_uv = (float *)malloc(image_size / 2 * sizeof(float));
    float* output_y = (float *)malloc(image_size * sizeof(float));
    float* output_uv = (float *)malloc(image_size / 2 * sizeof(float));

    for (GLuint i = 0; i < image_size * 3; i++) {
        input_bgr[i] = (float)input_image.data[i];
    }

    convert_bgr_to_nv12 (input_bgr, input_y, input_uv, image_w, image_h);
    uint8_t* tmp_nv12 = (uint8_t *)malloc(image_size * 3 / 2 * sizeof(uint8_t));

    int table_scale_size[2] = {8, 8};
    int geo_table_size = image_size / (table_scale_size[0] * table_scale_size[1]);
    float* geo_table = (float *)malloc(geo_table_size * 2 * sizeof(float));
    
    for (GLuint row = 0; row < image_h / table_scale_size[1]; row++) {
        for (GLuint col = 0; col < image_w / table_scale_size[0]; col++) {
            int index = row * image_w / table_scale_size[0] + col;

            geo_table[index * 2] = (image_w / table_scale_size[0] - col) * table_scale_size[0];
            geo_table[index * 2 + 1] = row * table_scale_size[1];
        }
    }

    CHECK();
    computeProgram = createComputeProgram(gComputeShader);
    CHECK();

    setupSSBufferObject(input0_ssbo, 0, (void *)input_y, image_size * sizeof(float));
    setupSSBufferObject(input1_ssbo, 1, (void *)input_uv, image_size / 2 * sizeof(float));
    setupSSBufferObject(input2_ssbo, 2, (void *)geo_table, geo_table_size * 2 * sizeof(float));
    setupSSBufferObject(input3_ssbo, 3, (void *)table_scale_size, 2 * sizeof(int));
    setupSSBufferObject(input4_ssbo, 4, (void *)&image_w, sizeof(int));
    setupSSBufferObject(input5_ssbo, 5, (void *)&image_h, sizeof(int));
    setupSSBufferObject(output0_ssbo, 6, NULL, image_size * sizeof(float));
    setupSSBufferObject(output1_ssbo, 7, NULL, image_size / 2 * sizeof(float));   

    CHECK();

    struct timeval start;
    struct timeval end;
    unsigned long diff = 0;
    gettimeofday(&start, NULL);

    float *pOut;
    for(int i = 0; i < 100; i++) {
        glUseProgram(computeProgram);
        glDispatchCompute(image_w / 16, image_h / 16, 1);   // arraySize/local_size_x
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
        tmp_nv12[i] = (uint8_t)pOut[i];
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
   
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output1_ssbo);
    pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, image_size / 2 * sizeof(float), GL_MAP_READ_BIT);
    for (GLuint i = 0; i < image_size / 2; i++) {
        tmp_nv12[i + image_size] = (uint8_t)pOut[i];
    }

    FILE *fp = fopen("output.nv12", "w");
    fwrite(tmp_nv12, 1, image_size * 3 / 2, fp);
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
