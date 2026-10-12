#!/bin/sh
# Makes an update of the image last built, as systemd-sysupdate takes one: its system's partition,
# compressed, and its UKI, named of its version, put into mkosi.output/updates beside the updates
# before it, of which the last two are kept; all listed in SHA256SUMS, signed by the box's update
# key, of keys.sh, which the image's /usr/lib/systemd/import-pubring.pgp checks it by.
#
#     box/image/publish.sh            # then serve mkosi.output/updates, as leatbox.com/updates does
#     box/image/publish.sh --upload   # and put it on leatbox.com/updates, for every box to take
#
# An update every box takes is of a product build, without the dev profile, of a commit.
set -eu
here=$(dirname "$(realpath "$0")")
out=$here/mkosi.output
keys=${LEAT_BOX_KEYS:-$HOME/.config/leat/box-keys}
updates=$out/updates
# the version, of the system's partition's name, leat_<version>, as the UKI boots it by
version=$(sfdisk -J "$out/leat-box.raw" | python3 -c '
import json, sys
names = [p.get("name", "") for p in json.load(sys.stdin)["partitiontable"]["partitions"]]
print(next(n.removeprefix("leat_") for n in names if n.startswith("leat_")))')
mkdir -p "$updates"
zstd -T0 -q -f "$out/leat-box.root-x86-64.raw" -o "$updates/leat_$version.root.raw.zst"
cp "$out/leat-box.efi" "$updates/leat_$version.efi"
cd "$updates"
for old in $(ls leat_*.efi | sort -V | head -n -2); do
    rm -f "$old" "${old%.efi}.root.raw.zst"
done
sha256sum leat_* > SHA256SUMS
gpg --homedir "$keys/gnupg" --batch --yes --detach-sign --output SHA256SUMS.gpg SHA256SUMS
echo "the update $version, in $updates"
[ "${1:-}" = --upload ] || exit 0
case $version in *-dirty) echo "refused: $version is of changes not committed" >&2; exit 1 ;; esac
host=${LEAT_UPDATES_HOST:-leatbox}
# the files first, and the manifest and its signature last, so that no box reads a manifest of files
# not there yet; then the versions it no longer lists taken away
rsync -t --partial leat_* "$host:/srv/leatbox-updates/"
rsync -t SHA256SUMS SHA256SUMS.gpg "$host:/srv/leatbox-updates/"
ssh "$host" 'cd /srv/leatbox-updates && for f in leat_*; do grep -q " $f\$" SHA256SUMS || rm -f "$f"; done'
echo "uploaded to $host:/srv/leatbox-updates, as https://leatbox.com/updates/"
