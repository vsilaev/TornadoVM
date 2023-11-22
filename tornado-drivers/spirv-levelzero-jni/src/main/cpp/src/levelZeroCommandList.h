/*
 * MIT License
 *
 * Copyright (c) 2021, APT Group, Department of Computer Science,
 * The University of Manchester.
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
 /* Header for class uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList */
 #include <jni.h>

#ifndef _Included_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
#define _Included_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendLaunchKernel_native
 * Signature: (JJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeGroupDispatch;Luk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILjava/lang/Object;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendLaunchKernel_1native
        (JNIEnv *, jobject, jlong, jlong, jobject, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListClose_native
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListClose_1native
        (JNIEnv *, jobject, jlong);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_native
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;[BJJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1native
        (JNIEnv *, jobject, jlong, jobject, jbyteArray, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeChar
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;[CJJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeChar
        (JNIEnv *, jobject, jlong, jobject, jcharArray, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeShort
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;[SJJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeShort
        (JNIEnv *, jobject, jlong, jobject, jshortArray, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeInt
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;[IJJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeInt
        (JNIEnv *, jobject, jlong, jobject, jintArray, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeFloat
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;[FJJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeFloat
        (JNIEnv *, jobject, jlong, jobject, jfloatArray, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeDouble
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;[DJJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeDouble
        (JNIEnv *, jobject, jlong, jobject, jdoubleArray, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeLong
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;[JJJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeLong
        (JNIEnv *, jobject, jlong, jobject, jlongArray, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeOffHeap
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;JJJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeOffHeap
        (JNIEnv *, jobject, jlong, jobject, jlong, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeBack
 * Signature: (J[BLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;JJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeBack
        (JNIEnv *, jobject, jlong, jbyteArray, jobject, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeBackChar
 * Signature: (J[CLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;JJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeBackChar
        (JNIEnv *, jobject, jlong, jcharArray, jobject, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeBackShort
 * Signature: (J[SLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;JJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeBackShort
        (JNIEnv *, jobject, jlong, jshortArray, jobject, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeBackInt
 * Signature: (J[ILuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;JJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeBackInt
        (JNIEnv *, jobject, jlong, jintArray, jobject, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeBackFloat
 * Signature: (J[FLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;JJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeBackFloat
        (JNIEnv *, jobject, jlong, jfloatArray, jobject, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeBackDouble
 * Signature: (J[DLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;JJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeBackDouble
        (JNIEnv *, jobject, jlong, jdoubleArray, jobject, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeBackLong
 * Signature: (J[JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;JJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeBackLong
        (JNIEnv *, jobject, jlong, jlongArray, jobject, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeBackOffHeapSegment
 * Signature: (JJLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;JJJLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeBackOffHeapSegment
        (JNIEnv *, jobject, jlong, jlong, jobject, jlong, jlong, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendBarrier_native
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILjava/lang/Object;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendBarrier_1native
        (JNIEnv *, jobject, jlong, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryCopy_nativeBuffers
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;Luk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryCopy_1nativeBuffers
        (JNIEnv *, jobject, jlong, jobject, jobject, jint, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListReset_native
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListReset_1native
        (JNIEnv *, jobject, jlong);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendQueryKernelTimestamps_native
 * Signature: (JILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;Luk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;[ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;I[Luk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendQueryKernelTimestamps_1native
        (JNIEnv *, jobject, jlong, jint, jobject, jobject, jintArray, jobject, jint, jobjectArray);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendWriteGlobalTimestamp_native
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroByteBuffer;Luk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;ILuk/ac/manchester/tornado/drivers/spirv/levelzero/ZeEventHandle;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendWriteGlobalTimestamp_1native
        (JNIEnv *, jobject, jlong, jobject, jobject, jint, jobject);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemoryPrefetch_native
 * Signature: (JLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroBufferInteger;I)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemoryPrefetch_1native
        (JNIEnv *, jobject, jlong, jobject, jint);

/*
 * Class:     uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList
 * Method:    zeCommandListAppendMemAdvise_native
 * Signature: (JJLuk/ac/manchester/tornado/drivers/spirv/levelzero/LevelZeroBufferInteger;II)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_spirv_levelzero_LevelZeroCommandList_zeCommandListAppendMemAdvise_1native
        (JNIEnv *, jobject, jlong, jlong, jobject, jint, jint);

#ifdef __cplusplus
}
#endif
#endif
