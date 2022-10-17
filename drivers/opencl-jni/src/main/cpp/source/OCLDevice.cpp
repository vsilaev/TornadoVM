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
