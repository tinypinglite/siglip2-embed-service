#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
compiler=${CC:-cc}
case "$(uname -s)" in
  Darwin) shared_flag=-dynamiclib ;;
  *) shared_flag=-shared ;;
esac

# pkg-config output intentionally undergoes word splitting for compiler flags.
# shellcheck disable=SC2046
"$compiler" -std=c11 -O2 -fPIC "$shared_flag" -Wall -Wextra -Werror \
  $(pkg-config --cflags libturbojpeg) \
  "$repo_dir/src/siglip2_embed/jpeg_decode.c" \
  -o "$repo_dir/src/siglip2_embed/_jpeg_decode.so" \
  $(pkg-config --libs libturbojpeg)
