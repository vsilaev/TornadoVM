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
#include <cstring>
#include "OCLContext.h"
#include "ocl_log.h"
#include "global_vars.h"

#if _WIN32
#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)
#define posix_memalign_free _aligned_free
#else
#define posix_memalign_free free
#endif

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    clReleaseContext
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_clReleaseContext
(JNIEnv *env, jclass clazz, jlong context_id) {
    cl_int status = clReleaseContext((cl_context) context_id);
    LOG_OCL_AND_VALIDATE("clReleaseContext", status);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    clGetContextInfo
 * Signature: (JI[B)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_clGetContextInfo
(JNIEnv *env, jclass clazz, jlong context_id, jint param_name, jbyteArray array) {
    jbyte *value = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(array, NULL));
    int len = env->GetArrayLength(array);
    size_t return_size = 0;
    cl_int status = clGetContextInfo((cl_context) context_id, (cl_context_info) param_name, len, (void *) value, &return_size);
    LOG_OCL_AND_VALIDATE("clGetContextInfo", status);
    if (NULL != value) {
        env->ReleasePrimitiveArrayCritical(array, value, 0);
    }
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    clCreateCommandQueue
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_clCreateCommandQueue
(JNIEnv *env, jclass clazz, jlong context_id, jlong device_id, jlong properties) {
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue((cl_context) context_id, (cl_device_id) device_id, (cl_command_queue_properties) properties, &status);
    LOG_OCL_AND_VALIDATE("clCreateCommandQueue", status);
    return CL_SUCCESS == status ? (jlong) queue : (jlong) status;
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    allocateNativeMemory
 * Signature: (JJ)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_allocateNativeMemory
(JNIEnv *env, jclass clazz, jlong size, jlong alignment) {
    void *ptr;
    int rc = posix_memalign(&ptr, (size_t) alignment, (size_t) size);
    if (rc != 0) {
        printf("OpenCL off-heap memory allocation (posix_memalign) failed. Error value: %d.\n", rc);
        return NULL;
    } else {
        memset(ptr, 0, (size_t) size);
        return env->NewDirectByteBuffer(ptr, size);
    }
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    freeNativeMemory
 * Signature: (Ljava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_freeNativeMemory
(JNIEnv *env, jclass clazz, jobject buffer) {
	void *address = env->GetDirectBufferAddress(buffer);
	posix_memalign_free(address);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    createBuffer
 * Signature: (JJJJ)Luk/ac/manchester/tornado/drivers/opencl/OCLContext/OCLBufferResult;
 */
JNIEXPORT jobject JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_createBuffer
(JNIEnv *env, jclass clazz, jlong context_id, jlong flags, jlong size, jlong host_ptr) {
    cl_mem mem;
    cl_int status;
	if (host_ptr == 0) {
        mem = clCreateBuffer((cl_context) context_id, (cl_mem_flags) flags, (size_t) size, NULL, &status);
	} else {
	    mem = clCreateBuffer((cl_context) context_id, (cl_mem_flags) flags, (size_t) size, (void *) host_ptr, &status);
	}
	LOG_OCL_AND_VALIDATE("clCreateBuffer", status);
    return env->NewObject(JCL_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLCONTEXT_OCLBUFFERRESULT,
    		              JM_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLCONTEXT_OCLBUFFERRESULT_NEW,
						  (jlong) mem, (jlong) host_ptr, (jint) status);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    createSubBuffer
 * Signature: (JJI[B)J
 */
JNIEXPORT jlong JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_createSubBuffer
(JNIEnv *env, jclass clazz, jlong buffer, jlong flags, jint buffer_create_type, jbyteArray array) {
    jbyte *buffer_create_info = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(array, NULL));
    cl_int status;
    cl_mem mem = clCreateSubBuffer((cl_mem) buffer, (cl_mem_flags) flags, (cl_buffer_create_type) buffer_create_type, (void *) buffer_create_info, &status);
    LOG_OCL_AND_VALIDATE("clCreateSubBuffer", status);
    env->ReleasePrimitiveArrayCritical(array, buffer_create_info, 0);
    return (jlong) mem;
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    clReleaseMemObject
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_clReleaseMemObject
(JNIEnv *env, jclass clazz, jlong memobj) {
    cl_int status = clReleaseMemObject((cl_mem) memobj);
    LOG_OCL_AND_VALIDATE("clReleaseMemObject", status);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    clCreateProgramWithSource
 * Signature: (J[B[J)J
 */
JNIEXPORT jlong JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_clCreateProgramWithSource
(JNIEnv *env, jclass clazz, jlong context_id, jbyteArray javaSourceArray, jlongArray javaSizeArray) {
    jbyte *source = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(javaSourceArray, NULL));
    jlong *lengths = static_cast<jlong *>(env->GetPrimitiveArrayCritical(javaSizeArray, NULL));
    jsize numLengths = env->GetArrayLength(javaSizeArray);

    cl_int status = CL_INVALID_PROGRAM;
    cl_program program = clCreateProgramWithSource((cl_context) context_id, (cl_uint) numLengths, (const char **) &source, (size_t*) lengths, &status);
    LOG_OCL_AND_VALIDATE("clCreateProgramWithSource", status);
    if (NULL != lengths) {
        env->ReleasePrimitiveArrayCritical(javaSizeArray, lengths, 0);
    }
    if (NULL != source) {
        env->ReleasePrimitiveArrayCritical(javaSourceArray, source, 0);
    }
    return CL_SUCCESS == status ? (jlong) program : (jlong) status;
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    clCreateProgramWithBinary
 * Signature: (JJ[B[J)J
 */
JNIEXPORT jlong JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_clCreateProgramWithBinary
(JNIEnv *env, jclass clazz, jlong context_id, jlong device_id, jbyteArray javaSourceBinaryArray, jlongArray javaSizeArray) {
    jbyte *binary = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(javaSourceBinaryArray, NULL));
    jlong *lengths = static_cast<jlong *>(env->GetPrimitiveArrayCritical(javaSizeArray, NULL));
    jsize numLengths = env->GetArrayLength(javaSizeArray);

    cl_int status = CL_INVALID_PROGRAM;
    cl_program program;
    if (numLengths == 1) {
        cl_int binary_status;
        program = clCreateProgramWithBinary((cl_context) context_id, (cl_uint) numLengths, (const cl_device_id *) &device_id, (const size_t*) lengths, (const unsigned char **) &binary, &binary_status, &status);
        LOG_OCL_AND_VALIDATE("clCreateProgramWithBinary", status);
    } else {
        std::cout << "[TornadoVM JNI] OCL> loading multiple binaries not supported\n";
    }
    if (NULL != lengths) {
        env->ReleasePrimitiveArrayCritical(javaSizeArray, lengths, 0);
    }
    if (NULL != binary) {
        env->ReleasePrimitiveArrayCritical(javaSourceBinaryArray, binary, 0);
    }

    return CL_SUCCESS == status ? (jlong) program : (jlong) status;
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLContext
 * Method:    clCreateProgramWithIL
 * Signature: (J[B[J)J
 */
JNIEXPORT jlong JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLContext_clCreateProgramWithIL
        (JNIEnv *env, jclass clazz, jlong context_id, jbyteArray javaSourceBinaryArray, jlongArray javaSizeArray) {

    #if CL_TARGET_OPENCL_VERSION >= 210
        jbyte *source = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(javaSourceBinaryArray, NULL));
        jlong *lengths = static_cast<jlong *>(env->GetPrimitiveArrayCritical(javaSizeArray, NULL));
        size_t binarySize = lengths[0];
        cl_int status = CL_INVALID_PROGRAM;
        cl_program program = clCreateProgramWithIL((cl_context) context_id, (const void *) source, binarySize, &status);
        LOG_OCL_AND_VALIDATE("clCreateProgramWithIL", status);
        if (NULL != lengths) {
            env->ReleasePrimitiveArrayCritical(javaSizeArray, lengths, 0);
        }
        if (NULL != source) {
            env->ReleasePrimitiveArrayCritical(javaSourceBinaryArray, source, 0);
        }
        return CL_SUCCESS == status ? (jlong) program : (jlong) status;
    #else
        return -1;
    #endif
}
