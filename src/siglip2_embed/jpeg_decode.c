#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include <turbojpeg.h>

enum {
  SAKURA_JPEG_OK = 0,
  SAKURA_JPEG_INVALID = 1,
  SAKURA_JPEG_SCAN_LIMIT = 2,
  SAKURA_JPEG_OUTPUT_LIMIT = 3,
};

static int sakura_jpeg_is_sof(uint8_t marker) {
  switch (marker) {
    case 0xc0:
    case 0xc1:
    case 0xc2:
    case 0xc3:
    case 0xc5:
    case 0xc6:
    case 0xc7:
    case 0xc9:
    case 0xca:
    case 0xcb:
    case 0xcd:
    case 0xce:
    case 0xcf:
      return 1;
    default:
      return 0;
  }
}

static int sakura_jpeg_is_progressive_sof(uint8_t marker) {
  return marker == 0xc2 || marker == 0xc6 || marker == 0xca || marker == 0xce;
}

static int sakura_jpeg_next_marker(const uint8_t* data, size_t data_size,
                                   size_t* position, int in_entropy,
                                   uint8_t* marker) {
  size_t index = *position;

  if (!in_entropy) {
    if (index >= data_size || data[index++] != 0xff) return SAKURA_JPEG_INVALID;
    do {
      if (index >= data_size) return SAKURA_JPEG_INVALID;
      *marker = data[index++];
    } while (*marker == 0xff);
    if (*marker == 0x00) return SAKURA_JPEG_INVALID;
    *position = index;
    return SAKURA_JPEG_OK;
  }

  while (index < data_size) {
    if (data[index++] != 0xff) continue;
    do {
      if (index >= data_size) return SAKURA_JPEG_INVALID;
      *marker = data[index++];
    } while (*marker == 0xff);
    if (*marker == 0x00 || (*marker >= 0xd0 && *marker <= 0xd7)) continue;
    *position = index;
    return SAKURA_JPEG_OK;
  }
  return SAKURA_JPEG_INVALID;
}

static int sakura_jpeg_scan_layout(const uint8_t* data, size_t data_size,
                                   int max_scans, int* is_progressive) {
  size_t position = 0;
  uint8_t marker;
  int in_entropy = 0;
  int progressive = 0;
  int saw_sof = 0;
  int scans = 0;

  if (max_scans < 1 ||
      sakura_jpeg_next_marker(data, data_size, &position, 0, &marker) !=
          SAKURA_JPEG_OK ||
      marker != 0xd8) {
    return SAKURA_JPEG_INVALID;
  }

  for (;;) {
    size_t segment_length;

    if (sakura_jpeg_next_marker(data, data_size, &position, in_entropy,
                                &marker) != SAKURA_JPEG_OK) {
      return SAKURA_JPEG_INVALID;
    }
    in_entropy = 0;

    if (marker == 0xd9) {
      if (!saw_sof || (progressive && scans == 0)) return SAKURA_JPEG_INVALID;
      *is_progressive = progressive;
      return SAKURA_JPEG_OK;
    }
    if (marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd7)) {
      return SAKURA_JPEG_INVALID;
    }
    if (marker == 0x01) continue;
    if (marker < 0xc0 || data_size - position < 2) {
      return SAKURA_JPEG_INVALID;
    }

    segment_length = ((size_t)data[position] << 8) | data[position + 1];
    if (segment_length < 2 || segment_length > data_size - position) {
      return SAKURA_JPEG_INVALID;
    }

    if (sakura_jpeg_is_sof(marker)) {
      if (saw_sof) return SAKURA_JPEG_INVALID;
      saw_sof = 1;
      progressive = sakura_jpeg_is_progressive_sof(marker);
      if (!progressive) {
        *is_progressive = 0;
        return SAKURA_JPEG_OK;
      }
    }
    if (marker == 0xda) {
      if (!saw_sof || ++scans > max_scans) {
        return scans > max_scans ? SAKURA_JPEG_SCAN_LIMIT : SAKURA_JPEG_INVALID;
      }
      in_entropy = 1;
    }
    position += segment_length;
  }
}

static int sakura_jpeg_scaled_dimension(int dimension, tjscalingfactor factor,
                                        int* scaled_dimension) {
  uint64_t scaled;

  if (dimension < 1 || factor.num < 1 || factor.denom < 1 ||
      factor.num > factor.denom) {
    return SAKURA_JPEG_OUTPUT_LIMIT;
  }
  scaled = ((uint64_t)dimension * (uint64_t)factor.num +
            (uint64_t)factor.denom - 1) /
           (uint64_t)factor.denom;
  if (scaled == 0 || scaled > INT_MAX) return SAKURA_JPEG_OUTPUT_LIMIT;
  *scaled_dimension = (int)scaled;
  return SAKURA_JPEG_OK;
}

static int sakura_jpeg_select_scaled_size(int source_width, int source_height,
                                          int max_pixels, int* output_width,
                                          int* output_height) {
  const tjscalingfactor* factors;
  uint64_t best_pixels = 0;
  int factor_count;
  int index;

  if (max_pixels < 1) return SAKURA_JPEG_OUTPUT_LIMIT;
  factors = tjGetScalingFactors(&factor_count);
  if (factors == NULL || factor_count < 1) return SAKURA_JPEG_OUTPUT_LIMIT;

  for (index = 0; index < factor_count; ++index) {
    int scaled_width;
    int scaled_height;
    uint64_t scaled_pixels;

    if (sakura_jpeg_scaled_dimension(source_width, factors[index],
                                     &scaled_width) != SAKURA_JPEG_OK ||
        sakura_jpeg_scaled_dimension(source_height, factors[index],
                                     &scaled_height) != SAKURA_JPEG_OK) {
      continue;
    }
    scaled_pixels = (uint64_t)scaled_width * (uint64_t)scaled_height;
    if (scaled_pixels <= (uint64_t)max_pixels && scaled_pixels > best_pixels) {
      best_pixels = scaled_pixels;
      *output_width = scaled_width;
      *output_height = scaled_height;
    }
  }
  return best_pixels == 0 ? SAKURA_JPEG_OUTPUT_LIMIT : SAKURA_JPEG_OK;
}

int sakura_jpeg_get_features(const uint8_t* data, size_t data_size,
                             int max_scans, int max_decoded_pixels,
                             int* source_width, int* source_height,
                             int* decoded_width, int* decoded_height,
                             int* is_progressive) {
  tjhandle handle;
  int width;
  int height;
  int subsampling;
  int color_space;
  int progressive;
  int status;

  if (data == NULL || source_width == NULL || source_height == NULL ||
      decoded_width == NULL || decoded_height == NULL ||
      is_progressive == NULL || data_size > ULONG_MAX) {
    return SAKURA_JPEG_INVALID;
  }
  status = sakura_jpeg_scan_layout(data, data_size, max_scans, &progressive);
  if (status != SAKURA_JPEG_OK) return status;

  handle = tjInitDecompress();
  if (handle == NULL) return SAKURA_JPEG_INVALID;
  status = tjDecompressHeader3(handle, (const unsigned char*)data,
                               (unsigned long)data_size, &width, &height,
                               &subsampling, &color_space);
  tjDestroy(handle);
  if (status != 0 || width < 1 || height < 1) return SAKURA_JPEG_INVALID;

  status = sakura_jpeg_select_scaled_size(width, height, max_decoded_pixels,
                                           decoded_width, decoded_height);
  if (status != SAKURA_JPEG_OK) return status;
  *source_width = width;
  *source_height = height;
  *is_progressive = progressive;
  return SAKURA_JPEG_OK;
}

int sakura_jpeg_decode_rgb_scaled(const uint8_t* data, size_t data_size,
                                  int width, int height, uint8_t* output,
                                  size_t output_size) {
  tjhandle handle;
  size_t stride;
  size_t required_size;
  int status;

  if (data == NULL || output == NULL || data_size > ULONG_MAX || width < 1 ||
      height < 1 || width > INT_MAX / 3) {
    return SAKURA_JPEG_INVALID;
  }
  stride = (size_t)width * 3;
  if ((size_t)height > SIZE_MAX / stride) return SAKURA_JPEG_INVALID;
  required_size = stride * (size_t)height;
  if (output_size < required_size) return SAKURA_JPEG_INVALID;

  handle = tjInitDecompress();
  if (handle == NULL) return SAKURA_JPEG_INVALID;
  status = tjDecompress2(handle, (const unsigned char*)data,
                         (unsigned long)data_size, (unsigned char*)output,
                         width, (int)stride, height, TJPF_RGB,
                         TJFLAG_LIMITSCANS | TJFLAG_STOPONWARNING);
  tjDestroy(handle);
  return status == 0 ? SAKURA_JPEG_OK : SAKURA_JPEG_INVALID;
}
