/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class de_libalf_jni_JNINormalizer */

#ifndef _Included_de_libalf_jni_JNINormalizer
#define _Included_de_libalf_jni_JNINormalizer
#ifdef __cplusplus
extern "C" {
#endif
#undef de_libalf_jni_JNINormalizer_serialVersionUID
#define de_libalf_jni_JNINormalizer_serialVersionUID 1LL
#undef de_libalf_jni_JNINormalizer_serialVersionUID
#define de_libalf_jni_JNINormalizer_serialVersionUID 1LL
/*
 * Class:     de_libalf_jni_JNINormalizer
 * Method:    init
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_de_libalf_jni_JNINormalizer_init
  (JNIEnv *, jobject);

/*
 * Class:     de_libalf_jni_JNINormalizer
 * Method:    serialize
 * Signature: (J)[I
 */
JNIEXPORT jintArray JNICALL Java_de_libalf_jni_JNINormalizer_serialize
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_libalf_jni_JNINormalizer
 * Method:    deserialize
 * Signature: ([IJ)Z
 */
JNIEXPORT jboolean JNICALL Java_de_libalf_jni_JNINormalizer_deserialize
  (JNIEnv *, jobject, jintArray, jlong);

/*
 * Class:     de_libalf_jni_JNINormalizer
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_de_libalf_jni_JNINormalizer_destroy
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
