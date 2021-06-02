/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020, APT Group, Department of Computer Science,
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

#ifndef EXTERN
#define EXTERN extern
#endif

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

EXTERN JavaVM *jvm;
EXTERN jclass JCL_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLEVENT_CALLBACK;
EXTERN jmethodID JM_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLEVENT_CALLBACK_EXECUTE;
EXTERN jclass JCL_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLCONTEXT_OCLBUFFERRESULT;
EXTERN jmethodID JM_UK_AC_MANCHESTER_TORNADO_DRIVERS_OPENCL_OCLCONTEXT_OCLBUFFERRESULT_NEW;

#ifdef __cplusplus
}
#endif

#define JNI_EVENT_HANDLER(env, event, ... ) do {\
    JNIEnv* (env);\
    bool __attached__ = false;\
    {\
        int jniStatus = jvm->GetEnv((void **)&(env), JNI_VERSION_1_2);\
        if (jniStatus == JNI_EDETACHED) {\
            /* std::cout << "GetEnv: not attached to current thread" << std::endl; */\
            if (jvm->AttachCurrentThread((void **)&(env), NULL) == JNI_OK) {\
                /* std::cout << "Attached ok" << std::endl; */\
                __attached__ = true;\
            } else {\
                /* std::cout << "Failed to attach" << std::endl; */\
            }\
        } else if (jniStatus == JNI_OK) {\
            /* already in JNI thread */\
        } else {\
            /* ERRORS */ \
            /* if (jniStatus == JNI_EVERSION) {} */\
            /* std::cout << "GetEnv: version not supported" << std::endl; */\
        }\
    } {\
    __VA_ARGS__\
    }\
    if (__attached__) {\
        jvm->DetachCurrentThread();\
    }\
    cl_int status = clReleaseEvent(event);\
    LOG_OCL_AND_VALIDATE("clReleaseEvent", status);\
} while (0)
