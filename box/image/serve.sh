#!/bin/sh
# Serves the image last built, compressed by zstd, and install.sh, its address written in, to a
# machine on this network booted on a live system, as one is installed onto its internal disk. On
# the machine, as root: curl -s http://ADDRESS:PORT/install | bash. The compressed image is tested
# once made, and install.sh checks the disk it wrote against the image's SHA-256.
#
#     box/image/serve.sh [port]    # 8099 by default; Ctrl-C stops it
set -eu
here=$(dirname "$(realpath "$0")")
image=$here/mkosi.output/leat-box.raw
port=${1:-8099}
address=$(ip -4 route get 192.0.2.1 | sed -n 's/.* src \([0-9.]*\).*/\1/p')
[ -f "$image" ] || { echo "there is no image at $image: build one with mkosi" >&2; exit 1; }
served=$(mktemp -d)
trap 'rm -rf "$served"' EXIT
if [ ! -f "$image.zst" ] || [ "$image" -nt "$image.zst" ]; then
    echo "compressing the image..."
    zstd -T0 -3 -q -f "$image" -o "$image.zst"
    zstd -t -q "$image.zst" || { rm -f "$image.zst"; echo "the compressed image is corrupt" >&2; exit 1; }
    sha256sum "$image" | cut -d' ' -f1 > "$image.sha256"
fi
ln -s "$image.zst" "$served/leat-box.raw.zst"
sed -e "s|@URL@|http://$address:$port|" -e "s|@SIZE@|$(stat -c %s "$image")|" \
    -e "s|@SHA256@|$(cat "$image.sha256")|" "$here/install.sh" > "$served/install"
echo "on the machine, as root: curl -s http://$address:$port/install | bash"
python3 -m http.server "$port" --bind "$address" --directory "$served"
