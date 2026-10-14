#include <jni.h>

#include "videogammetry/version.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_videogammetry_NativeBridge_version(JNIEnv* env, jclass) {
    return env->NewStringUTF(vg::version());
}
