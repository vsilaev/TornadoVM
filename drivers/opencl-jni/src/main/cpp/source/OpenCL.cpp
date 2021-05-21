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

    GET_CLASS_METHOD(JCL_JAVA_UTIL_CONSUMER, "java/util/function/Consumer", 
                     JM_JAVA_UTIL_CONSUMER_ACCEPT, "accept", "(Ljava/lang/Object;)V");

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
        if (JCL_JAVA_UTIL_CONSUMER != NULL) {
            env->DeleteGlobalRef(JCL_JAVA_UTIL_CONSUMER);
        }
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
