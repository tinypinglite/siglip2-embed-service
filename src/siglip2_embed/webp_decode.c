#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include <webp/decode.h>

int sakura_webp_get_features(const uint8_t* data, size_t data_size,
                             int* width, int* height, int* has_animation) {
  WebPBitstreamFeatures features;
  const VP8StatusCode status = WebPGetFeatures(data, data_size, &features);
  if (status != VP8_STATUS_OK) return status;
  *width = features.width;
  *height = features.height;
  *has_animation = features.has_animation;
  return VP8_STATUS_OK;
}

int sakura_webp_decode_rgb_scaled(const uint8_t* data, size_t data_size,
                                  int width, int height, uint8_t* output,
                                  size_t output_size) {
  WebPDecoderConfig config;
  size_t stride;
  size_t required_size;
  VP8StatusCode status;

  if (data == NULL || output == NULL || width < 1 || height < 1 ||
      width > INT_MAX / 3) {
    return VP8_STATUS_INVALID_PARAM;
  }
  stride = (size_t)width * 3;
  if ((size_t)height > SIZE_MAX / stride) return VP8_STATUS_INVALID_PARAM;
  required_size = stride * (size_t)height;
  if (output_size < required_size) return VP8_STATUS_INVALID_PARAM;

  if (!WebPInitDecoderConfig(&config)) return -1;
  status = WebPGetFeatures(data, data_size, &config.input);
  if (status != VP8_STATUS_OK) return status;
  if (config.input.has_animation) return VP8_STATUS_UNSUPPORTED_FEATURE;

  config.output.colorspace = MODE_RGB;
  config.output.is_external_memory = 1;
  config.output.u.RGBA.rgba = output;
  config.output.u.RGBA.stride = (int)stride;
  config.output.u.RGBA.size = output_size;
  config.options.use_scaling = 1;
  config.options.scaled_width = width;
  config.options.scaled_height = height;
  config.options.use_threads = 0;

  status = WebPDecode(data, data_size, &config);
  WebPFreeDecBuffer(&config.output);
  return status;
}
