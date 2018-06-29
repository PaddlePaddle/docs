/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License */

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>
#include <string.h>
#include <memory>
#include "binary_reader.h"
#include "paddle_image_recognizer.h"

extern "C" {

JNIEXPORT jlong
Java_com_paddlepaddle_pdcamera_ImageRecognizer_init(JNIEnv *env,
                                                    jobject thiz,
                                                    jobject jasset_manager,
                                                    jstring jmerged_model_path,
                                                    jint jnormed_height,
                                                    jint jnormed_width,
                                                    jint jnormed_channel,
                                                    jfloatArray jmeans) {
  if (jmerged_model_path == nullptr) {
    return 0;
  }

  AAssetManager *aasset_manager = AAssetManager_fromJava(env, jasset_manager);
  BinaryReader::set_aasset_manager(aasset_manager);

  const char *merged_model_path = env->GetStringUTFChars(jmerged_model_path, 0);

  const float *m = env->GetFloatArrayElements(jmeans, 0);
  std::vector<float> means(m, m + sizeof(m) / sizeof(m[0]));

  ImageRecognizer *recognizer = new ImageRecognizer();
  recognizer->init(
      merged_model_path, jnormed_height, jnormed_width, jnormed_channel, means);

  env->ReleaseStringUTFChars(jmerged_model_path, merged_model_path);

  return reinterpret_cast<jlong>(recognizer);
}

JNIEXPORT jfloatArray
Java_com_paddlepaddle_pdcamera_ImageRecognizer_infer(JNIEnv *env,
                                                     jobject thiz,
                                                     jlong jrecognizer,
                                                     jbyteArray jpixels,
                                                     jint jheight,
                                                     jint jwidth,
                                                     jint jchannel,
                                                     jboolean jback_camera) {
  if (jrecognizer == 0 || jpixels == nullptr || jheight <= 0 || jwidth <= 0 ||
      jchannel <= 0) {
    return nullptr;
  }

  ImageRecognizer *recognizer =
      reinterpret_cast<ImageRecognizer *>(jrecognizer);

  ImageRecognizer::Result result;
  image::Config config(
      image::kRGB,
      jback_camera == true ? image::CLOCKWISE_R90 : image::CLOCKWISE_R270);

  const unsigned char *pixels =
      (unsigned char *)env->GetByteArrayElements(jpixels, 0);
  recognizer->infer(pixels, jheight, jwidth, jchannel, config, result);
  env->ReleaseByteArrayElements(jpixels, (jbyte *)pixels, 0);

  uint64_t result_size = result.height * result.width;
  jfloatArray jresult = env->NewFloatArray(result_size);
  jfloat *jresult_ptr = env->GetFloatArrayElements(jresult, 0);
  memcpy(jresult_ptr, result.data, result_size * sizeof(float));
  env->ReleaseFloatArrayElements(jresult, jresult_ptr, 0);

  return jresult;
}

JNIEXPORT void Java_com_paddlepaddle_pdcamera_ImageRecognizer_release(
    JNIEnv *env, jobject thiz, jlong jrecognizer) {
  if (jrecognizer == 0) {
    return;
  }

  ImageRecognizer *recognizer =
      reinterpret_cast<ImageRecognizer *>(jrecognizer);
  recognizer->release();
}
}
