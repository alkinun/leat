#!/bin/sh
# Makes an update of the image last built, as systemd-sysupdate takes one: its system's partition,
# compressed, and its UKI, named of its version, put into mkosi.output/updates/<channel> beside the
# updates before it, of which the last two are kept; all listed in SHA256SUMS, signed by the box's
# update key, of keys.sh, which the image's /usr/lib/systemd/import-pubring.pgp checks it by.
#
# Its channel is the image's own, as mkosi.postinst.chroot marked it: stable, of a build without the
# dev profile, which every box takes from leatbox.com/updates; or dev, of one with it, which boxes
# under test take from leatbox.com/updates/dev, as a box takes no dev image, nor a box under test
# a box's, which has no SSH.
#
#     box/image/publish.sh            # then serve mkosi.output/updates/<channel>, as leatbox.com does
#     box/image/publish.sh --upload   # and put it on leatbox.com, for the boxes of its channel
set -eu
here=$(dirname "$(realpath "$0")")
out=$here/mkosi.output
keys=${LEAT_BOX_KEYS:-$HOME/.config/leat/box-keys}
# the version, of the system's partition's name, leat_<version>, as the UKI boots it by
version=$(sfdisk -J "$out/leat-box.raw" | python3 -c '
import json, sys
names = [p.get("name", "") for p in json.load(sys.stdin)["partitiontable"]["partitions"]]
print(next(n.removeprefix("leat_") for n in names if n.startswith("leat_")))')
# the channel, as the system's partition says it
channel=$(debugfs -R "cat /usr/lib/leat/channel" "$out/leat-box.root-x86-64.raw" 2>/dev/null)
case $channel in
stable) path= ;;
dev) path=dev/ ;;
*) echo "refused: the image marks no channel: build it again" >&2; exit 1 ;;
esac
updates=$out/updates/$channel
mkdir -p "$updates"
zstd -T0 -q -f "$out/leat-box.root-x86-64.raw" -o "$updates/leat_$version.root.raw.zst"
cp "$out/leat-box.efi" "$updates/leat_$version.efi"
cd "$updates"
for old in $(ls leat_*.efi | sort -V | head -n -2); do
    rm -f "$old" "${old%.efi}.root.raw.zst"
done
sha256sum leat_* > SHA256SUMS
gpg --homedir "$keys/gnupg" --batch --yes --detach-sign --output SHA256SUMS.gpg SHA256SUMS
gpgconf --homedir "$keys/gnupg" --kill gpg-agent  # which signing started, and which outlives it else
echo "the $channel update $version, in $updates"
[ "${1:-}" = --upload ] || exit 0
case $version in *-dirty) echo "refused: $version is of changes not committed" >&2; exit 1 ;; esac
host=${LEAT_UPDATES_HOST:-leatbox}
target=/srv/leatbox-updates/$path
# the files first, and the manifest and its signature last, so that no box reads a manifest of files
# not there yet; then the versions it no longer lists taken away
ssh "$host" "mkdir -p $target"
rsync -t --partial leat_* "$host:$target"
rsync -t SHA256SUMS SHA256SUMS.gpg "$host:$target"
ssh "$host" "cd $target && for f in leat_*; do grep -q \" \$f\\\$\" SHA256SUMS || rm -f \"\$f\"; done"
echo "uploaded to $host:$target, as https://leatbox.com/updates/$path"
