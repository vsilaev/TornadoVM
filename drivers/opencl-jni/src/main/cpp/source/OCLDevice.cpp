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
 *
 */
#include <jni.h>

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <iostream>
#include <stdio.h>
#include "OCLDevice.h"
#include "ocl_log.h"

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLDevice
 * Method:    clGetDeviceInfo
 * Signature: (JI[B)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLDevice_clGetDeviceInfo__JI_3B
(JNIEnv *env, jclass clazz, jlong device_id, jint device_info, jbyteArray array) {
    jbyte *value = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(array, 0));
    jsize len = env->GetArrayLength(array);

    if (LOG_JNI) {
        std::cout << "size of cl_device_info: " << sizeof(cl_device_info) << std::endl;
        std::cout << "param_name: " <<  device_info << std::endl;
        std::cout << "len: " << len << std::endl;
    }

    size_t return_size = 0;
    cl_int status = clGetDeviceInfo((cl_device_id) device_id, (cl_device_info) device_info, len, (void *) value, &return_size);
    LOG_OCL_AND_VALIDATE("clGetDeviceInfo", status);
    if (NULL != value) {
        env->ReleasePrimitiveArrayCritical(array, value, 0);
    }
}


/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLDevice
 * Method:    clGetDeviceInfo
 * Signature: (JI)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLDevice_clGetDeviceInfo__JI
(JNIEnv *env, jclass clazz, jlong device_id, jint device_info) {
    if (LOG_JNI) {
        std::cout << "size of cl_device_info: " << sizeof(cl_device_info) << std::endl;
        std::cout << "param_name: " <<  device_info << std::endl;
    }

    size_t return_size = 0;
    cl_int status = clGetDeviceInfo((cl_device_id) device_id, (cl_device_info) device_info, 0, NULL, &return_size);
    LOG_OCL_AND_VALIDATE("clGetDeviceInfo-size", status);
    if (status != CL_SUCCESS || return_size < 1) {
        return NULL;
    }
    void* value = malloc(return_size);
    memset(value, 0, return_size);
    status = clGetDeviceInfo((cl_device_id) device_id, (cl_device_info) device_info, return_size, value, NULL);
    LOG_OCL_AND_VALIDATE("clGetDeviceInfo", status);
    if (status == CL_SUCCESS) {
        return env->NewDirectByteBuffer(value, return_size);
    } else {
        return NULL;
    }
}
