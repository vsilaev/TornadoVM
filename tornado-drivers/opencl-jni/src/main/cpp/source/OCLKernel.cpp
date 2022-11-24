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
#include "OCLKernel.h"
#include "ocl_log.h"

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLKernel
 * Method:    clReleaseKernel
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLKernel_clReleaseKernel
(JNIEnv *env, jclass clazz, jlong kernel_id) {
   cl_int status = clReleaseKernel((cl_kernel) kernel_id);
   LOG_OCL_AND_VALIDATE("clReleaseKernel", status);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLKernel
 * Method:    clSetKernelArgArray
 * Signature: (JIJ[B)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLKernel_clSetKernelArgArray
(JNIEnv *env, jclass clazz, jlong kernel_id, jint index, jlong size, jbyteArray array) {
    jbyte *value = static_cast<jbyte *>((array == NULL) ? NULL : env->GetPrimitiveArrayCritical(array, 0));
    cl_int status = clSetKernelArg((cl_kernel) kernel_id, (cl_uint) index, (size_t) size, (void*) value);
    LOG_OCL_AND_VALIDATE("clSetKernelArg", status);
    if (value != NULL) {
        env->ReleasePrimitiveArrayCritical(array, value, 0);
    }
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLKernel
 * Method:    clSetKernelArgBuffer
 * Signature: (JIJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLKernel_clSetKernelArgBuffer
(JNIEnv *env, jclass clazz, jlong kernel_id, jint index, jlong size, jobject buffer) {
    void *value = buffer == NULL ? NULL : env->GetDirectBufferAddress(buffer);
    cl_int status = clSetKernelArg((cl_kernel) kernel_id, (cl_uint) index, (size_t) size, value);
    LOG_OCL_AND_VALIDATE("clSetKernelArg", status);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLKernel
 * Method:    clGetKernelInfo
 * Signature: (JI[B)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLKernel_clGetKernelInfo__JI_3B
(JNIEnv *env, jclass clazz, jlong kernel_id, jint kernel_info, jbyteArray array) {
    jbyte *value;
    jsize len;
    value = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(array, 0));
    len = env->GetArrayLength(array);
    size_t return_size = 0;
    cl_int status = clGetKernelInfo((cl_kernel) kernel_id, (cl_kernel_info) kernel_info, len, (void *) value, &return_size);
    LOG_OCL_AND_VALIDATE("clGetKernelInfo", status);
    if (NULL != value) {
        env->ReleasePrimitiveArrayCritical(array, value, 0);
    }
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLKernel
 * Method:    clGetKernelInfo
 * Signature: (JI)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLKernel_clGetKernelInfo__JI
(JNIEnv *env, jclass clazz, jlong kernel_id, jint kernel_info) {
    size_t return_size = 0;
    cl_int status = clGetKernelInfo((cl_kernel) kernel_id, (cl_kernel_info) kernel_info, 0, NULL, &return_size);
    LOG_OCL_AND_VALIDATE("clGetKernelInfo-size", status);
    if (status != CL_SUCCESS || return_size < 1) {
        return NULL;
    }
    void* value = malloc(return_size);
    memset(value, 0, return_size);
    status = clGetKernelInfo((cl_kernel) kernel_id, (cl_kernel_info) kernel_info, return_size, value, NULL);
    LOG_OCL_AND_VALIDATE("clGetKernelInfo", status);
    if (status == CL_SUCCESS) {
        return env->NewDirectByteBuffer(value, return_size);
    } else {
        return NULL;
    }
}
