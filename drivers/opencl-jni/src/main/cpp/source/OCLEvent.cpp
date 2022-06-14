/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
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

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>

#endif

#include <iostream>
#include "OCLEvent.h"
#include "ocl_log.h"
#include "global_vars.h"
#include "utils.h"
/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLEvent
 * Method:    clGetEventInfo
 * Signature: (JI[B)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLEvent_clGetEventInfo
        (JNIEnv *env, jclass clazz, jlong event_id, jint param_name, jbyteArray array) {
    jbyte *value;
    jsize len;
    value = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(array, NULL));
    len = env->GetArrayLength(array);
    size_t return_size = 0;
    cl_int status = clGetEventInfo((cl_event) event_id, (cl_event_info) param_name, len, (void *) value, &return_size);
    LOG_OCL_AND_VALIDATE("clGetEventInfo", status);
    env->ReleasePrimitiveArrayCritical(array, value, 0);
}

/*
 * Class:     jacc_runtime_drivers_opencl_OCLEvent
 * Method:    clGetEventProfilingInfo
 * Signature: (JJ[B)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLEvent_clGetEventProfilingInfo
        (JNIEnv *env, jclass clazz, jlong event_id, jlong param_name, jbyteArray array) {
    jbyte *value = static_cast<jbyte *>(env->GetPrimitiveArrayCritical(array, NULL));
    size_t return_size = 0;
    cl_int status = clGetEventProfilingInfo((cl_event) event_id, (cl_profiling_info) param_name, sizeof(cl_ulong),
                                            (void *) value, NULL);
    LOG_OCL_AND_VALIDATE("clGetEventProfilingInfo", status);
    env->ReleasePrimitiveArrayCritical(array, value, 0);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLEvent
 * Method:    clWaitForEvents
 * Signature: ([J)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLEvent_clWaitForEvents
        (JNIEnv *env, jclass clazz, jlongArray array) {
    if (array != NULL) {
        jsize len;
        cl_event *events = static_cast<cl_event *>(env->GetPrimitiveArrayCritical(array, NULL));
        len = env->GetArrayLength(array);
        cl_int status = clWaitForEvents((cl_uint) len, (const cl_event *) events);
        LOG_OCL_AND_VALIDATE("clWaitForEvents", status);
        env->ReleasePrimitiveArrayCritical(array, events, 0);
    }
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLEvent
 * Method:    clReleaseEvent
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLEvent_clReleaseEvent
        (JNIEnv *env, jclass clazz, jlong event) {
    cl_int status = clReleaseEvent((const cl_event) event);
    LOG_OCL_AND_VALIDATE("clReleaseEvent", status);
}

void CL_CALLBACK onOpenCLEventCompletion(cl_event event, cl_int eventStatus, void *user_data);

/*
 * Class:     uk_ac_manchester_tornado_drivers_opencl_OCLEvent
 * Method:    clAttachCallback
 * Signature: (JLuk/ac/manchester/tornado/drivers/opencl/CLEvent$Callback;)V
 */
JNIEXPORT void JNICALL Java_uk_ac_manchester_tornado_drivers_opencl_OCLEvent_clAttachCallback
        (JNIEnv *env, jclass clazz, jlong event, jobject callback) {
    jobject sharedCallback = env->NewGlobalRef(callback);
    cl_int status;
    status = clRetainEvent((const cl_event) event); 
    LOG_OCL_AND_VALIDATE("clRetainEvent", status);
    status = clSetEventCallback((const cl_event) event, CL_COMPLETE, &onOpenCLEventCompletion, sharedCallback);
    LOG_OCL_AND_VALIDATE("clSetEventCallback", status);
}

void CL_CALLBACK onOpenCLEventCompletion(cl_event event, cl_int eventStatus, void *user_data) {
    JNI_EVENT_HANDLER(env, event, {
        jobject target = static_cast<jobject>(user_data); 
        env->CallVoidMethod(target, 
                            JM_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLEVENT_CALLBACK_EXECUTE, 
                            event, eventStatus);
        env->DeleteGlobalRef(target);
    });
}
