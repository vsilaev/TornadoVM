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

#define CL_TARGET_OPENCL_VERSION 120
#define EXTERN

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <iostream>
#include <stdio.h>
#include "OpenCL.h"
#include "ocl_log.h"
#include "global_vars.h"

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    jint INVALID_RESULT = -1;

    jvm = vm;

    JNIEnv *env;
    int jniStatus = jvm->GetEnv((void **)&env, JNI_VERSION_1_2);
    if (jniStatus == JNI_EDETACHED) {
        std::cout << "Unable to get JNI environment" << std::endl;
        return INVALID_RESULT;
    }
    jclass clazz;

#define GET_CLASS_METHOD(classVar, className, methodVar, methodName, methodSignature) do { \
    jclass clazz = env->FindClass((className)); \
    if (clazz != NULL) { \
        (classVar) = static_cast<jclass>(env->NewGlobalRef(clazz)); \
        jmethodID method = env->GetMethodID(clazz, (methodName), (methodSignature)); \
        if (method != NULL) { \
            (methodVar) = method; \
        } else { \
            std::cout << "Unable to get method " << (methodName) << (methodSignature) << " of class " << (className) << std::endl; \
            env->DeleteGlobalRef((classVar)); \
            (classVar) = NULL; \
            return INVALID_RESULT; \
        } \
    } else { \
        std::cout << "Unable to get class " << (className) << std::endl; \
        return INVALID_RESULT; \
    } \
} while (0)

    GET_CLASS_METHOD(JCL_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLEVENT_CALLBACK, "uk/ac/manchester/tornado/drivers/opencl/OCLEvent$Callback", 
                     JM_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLEVENT_CALLBACK_EXECUTE, "execute", "(JI)V");

    GET_CLASS_METHOD(JCL_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLCONTEXT_OCLBUFFERRESULT, "uk/ac/manchester/tornado/drivers/opencl/OCLContext$OCLBufferResult",
                     JM_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLCONTEXT_OCLBUFFERRESULT_NEW, "<init>", "(JJI)V");

    return JNI_VERSION_1_2;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv *env;
    int jniStatus = jvm->GetEnv((void **)&env, JNI_VERSION_1_2);
    if (env != NULL) {
        if (JCL_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLEVENT_CALLBACK != NULL) {
            env->DeleteGlobalRef(JCL_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLEVENT_CALLBACK);
        }
        if (JCL_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLCONTEXT_OCLBUFFERRESULT != NULL) {
            env->DeleteGlobalRef(JCL_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLCONTEXT_OCLBUFFERRESULT);
        }
    }
    jvm = NULL;
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OpenCL
 * Method:    clGetPlatformCount
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OpenCL_clGetPlatformCount
(JNIEnv *env, jclass clazz) {
    cl_uint num_platforms = 0;
    cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
    LOG_OCL_AND_VALIDATE("clGetPlatformIDs", status);
    return (jint) num_platforms;
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OpenCL
 * Method:    clGetPlatformIDs
 * Signature: ([J)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OpenCL_clGetPlatformIDs
(JNIEnv *env, jclass clazz, jlongArray array) {
    jlong *platforms;
    jsize len;

    platforms = static_cast<jlong *>(env->GetPrimitiveArrayCritical(array, 0));
    len = env->GetArrayLength(array);

    cl_uint num_platforms = 0;
    cl_int status = clGetPlatformIDs(len, (cl_platform_id*) platforms, &num_platforms);
    LOG_OCL_AND_VALIDATE("clGetPlatformIDs", status);
    env->ReleasePrimitiveArrayCritical(array, platforms, 0);
    return (jint) num_platforms;
}
