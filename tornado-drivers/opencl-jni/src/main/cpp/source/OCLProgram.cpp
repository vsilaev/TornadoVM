/*
 * MIT License
 *
 * Copyright (c) 2020-2022, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * Copyright (c) 2013-2020, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <jni.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include "OCLProgram.h"
#include "ocl_log.h"

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLProgram
 * Method:    clReleaseProgram
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLProgram_clReleaseProgram
(JNIEnv *env, jclass clazz, jlong program_id) {
    cl_int status = clReleaseProgram((cl_program) program_id);
    LOG_OCL_AND_VALIDATE("clReleaseProgram", status);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLProgram
 * Method:    clBuildProgram
 * Signature: (J[J[C)J
 */
JNIEXPORT jlong JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLProgram_clBuildProgram
(JNIEnv *env, jclass clazz, jlong program_id, jlongArray array1, jstring str) {
    jlong *devices = static_cast<jlong *>(env->GetPrimitiveArrayCritical(array1, NULL));
    jsize numDevices = env->GetArrayLength(array1);
    const char *options = env->GetStringUTFChars(str, NULL);

    cl_int status = clBuildProgram((cl_program) program_id, (cl_uint) numDevices, (cl_device_id*) devices, options, NULL, NULL);

    if (NULL != options) {
    	env->ReleaseStringUTFChars(str, options);
    }

    if (NULL != devices) {
        env->ReleasePrimitiveArrayCritical(array1, devices, 0);
    }
    LOG_OCL_AND_VALIDATE("clBuildProgram", status);
    return (jlong)status;

}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLProgram
 * Method:    clGetProgramInfo
 * Signature: (JI[B)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLProgram_clGetProgramInfo__JI_3B
(JNIEnv *env, jclass clazz, jlong program_id, jint param_name, jbyteArray array) {
    jbyte *value = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(array, NULL));
    jsize len = env->GetArrayLength(array);

    if (LOG_JNI) {
        std::cout << "size of cl_program_info: " << sizeof(cl_program_info) << std::endl;
        std::cout << "param_name: " <<  param_name << std::endl;
        std::cout << "len: " << len << std::endl;
    }
    size_t return_size = 0;
    cl_int status = clGetProgramInfo((cl_program) program_id, (cl_program_info) param_name, len, (void *) value, &return_size);
    LOG_OCL_AND_VALIDATE("clGetProgramInfo", status);
    if (NULL != value) {
        env->ReleasePrimitiveArrayCritical(array, value, 0);
    }
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLProgram
 * Method:    clGetProgramInfo
 * Signature: (JI)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLProgram_clGetProgramInfo__JI
(JNIEnv *env, jclass clazz, jlong program_id, jint param_name) {
    if (LOG_JNI) {
        std::cout << "size of cl_program_info: " << sizeof(cl_program_info) << std::endl;
        std::cout << "param_name: " <<  param_name << std::endl;
    }
    size_t return_size = 0;
    cl_int status = clGetProgramInfo((cl_program) program_id, (cl_program_info) param_name, 0, NULL, &return_size);
    LOG_OCL_AND_VALIDATE("clGetProgramInfo-size", status);
    if (status != CL_SUCCESS || return_size < 1) {
        return NULL;
    }
    void* value = malloc(return_size);
    memset(value, 0, return_size);
    status = clGetProgramInfo((cl_program) program_id, (cl_program_info) param_name, return_size, value, NULL);
    LOG_OCL_AND_VALIDATE("clGetProgramInfo", status);
    if (status == CL_SUCCESS) {
        return env->NewDirectByteBuffer(value, return_size);
    } else {
        return NULL;
    }
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLProgram
 * Method:    clGetProgramBuildInfo
 * Signature: (JJI[B)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLProgram_clGetProgramBuildInfo__JJI_3B
(JNIEnv *env, jclass clazz, jlong program_id, jlong device_id, jint param_name, jbyteArray array) {
    jbyte *value = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(array, NULL));
    jsize len = env->GetArrayLength(array);
    size_t return_size = 0;
    cl_int status = clGetProgramBuildInfo((cl_program) program_id, (cl_device_id) device_id, (cl_program_build_info) param_name, len, (void *) value, &return_size);
    LOG_OCL_AND_VALIDATE("clGetProgramBuildInfo", status);
    if (NULL != value) {
        env->ReleasePrimitiveArrayCritical(array, value, 0);
    }
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLProgram
 * Method:    clGetProgramBuildInfo
 * Signature: (JJI)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLProgram_clGetProgramBuildInfo__JJI
(JNIEnv *env, jclass clazz, jlong program_id, jlong device_id, jint param_name) {
    size_t return_size = 0;
    cl_int status = clGetProgramBuildInfo((cl_program) program_id, (cl_device_id) device_id, (cl_program_build_info) param_name, 0, NULL, &return_size);
    LOG_OCL_AND_VALIDATE("clGetProgramBuildInfo-size", status);
    if (status != CL_SUCCESS || return_size < 1) {
        return NULL;
    }
    void* value = malloc(return_size);
    memset(value, 0, return_size);
    status = clGetProgramBuildInfo((cl_program) program_id, (cl_device_id) device_id, (cl_program_build_info) param_name, return_size, value, NULL);
    LOG_OCL_AND_VALIDATE("clGetProgramBuildInfo", status);
    if (status == CL_SUCCESS) {
        return env->NewDirectByteBuffer(value, return_size);
    } else {
        return NULL;
    }
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLProgram
 * Method:    clCreateKernel
 * Signature: (JLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLProgram_clCreateKernel
(JNIEnv *env, jclass clazz, jlong program_id, jstring str) {
    const char *kernel_name = env->GetStringUTFChars(str, NULL);
    cl_int status = CL_INVALID_KERNEL;
    cl_kernel kernel = clCreateKernel((cl_program) program_id, kernel_name, &status);
    if (NULL != kernel_name) {
    	env->ReleaseStringUTFChars(str, kernel_name);
    }
    LOG_OCL_AND_VALIDATE("clCreateKernel", status);
    return status == CL_SUCCESS ? (jlong) kernel : (jlong)status;
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLProgram
 * Method:    getBinaries
 * Signature: (JJ[B)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLProgram_getBinaries
(JNIEnv *env, jclass clazz, jlong program_id, jlong num_devices, jobject array) {
    jbyte *value = (jbyte *) env->GetDirectBufferAddress(array);

    size_t return_size = 0;
    size_t *binarySizes = static_cast<size_t *>(malloc(sizeof(size_t) * num_devices));
    cl_int status = clGetProgramInfo((cl_program) program_id, CL_PROGRAM_BINARY_SIZES, sizeof (size_t) * num_devices, binarySizes, &return_size);
    LOG_OCL_AND_VALIDATE("clGetProgramInfo", status);

    jbyte **binaries = static_cast<jbyte **>(malloc(sizeof(unsigned char *) * num_devices));
    binaries[0] = value;
    for (int i = 1; i < num_devices; i++) {
        binaries[i] = value + binarySizes[i - 1];
    }
    status = clGetProgramInfo((cl_program) program_id, CL_PROGRAM_BINARIES, sizeof (unsigned char**), (void *) binaries, &return_size);
    LOG_OCL_AND_VALIDATE("clGetProgramInfo", status);
    free(binarySizes);
    free(binaries);
}
