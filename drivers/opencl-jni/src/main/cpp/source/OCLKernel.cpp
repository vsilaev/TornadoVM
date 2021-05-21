/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
#include <jni.h>

#define CL_TARGET_OPENCL_VERSION 120
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
 * Method:    clSetKernelArg
 * Signature: (JIJ[B)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLKernel_clSetKernelArg
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
