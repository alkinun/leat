#!/bin/sh
# Writes the image to a USB stick, a box under test's, and moves a box's registration onto its boot
# partition, for the box to take in on its first boot. Refuses any disk but a whole USB or removable
# one, not mounted, of 20 GB at least, as the box's first boot lays out 18.5 GB; asks before it
# erases it. The registration is deleted here once on the stick.
#
#     ssh leatbox leatbox-registry register > leat-box.json
#     sudo box/image/flash.sh /dev/sdX leat-box.json
set -eu
image=$(dirname "$(realpath "$0")")/mkosi.output/leat-box.raw
disk=${1:?give the stick, as /dev/sdb}
registration=${2:-}
refuse() { echo "refused: $1" >&2; exit 1; }

[ -f "$image" ] || refuse "there is no image at $image: build one with mkosi"
[ -z "$registration" ] || [ -f "$registration" ] || refuse "there is no registration at $registration"
[ -b "$disk" ] && [ "$(lsblk -dno TYPE "$disk")" = disk ] || refuse "$disk is not a whole disk"
[ "$(lsblk -dno TRAN "$disk")" = usb ] || [ "$(lsblk -dno RM "$disk")" = 1 ] \
    || refuse "$disk is neither a USB nor a removable disk"
[ -z "$(lsblk -no MOUNTPOINTS "$disk" | tr -d '[:space:]')" ] || refuse "$disk has a partition mounted"
size=$(lsblk -bdno SIZE "$disk")
[ "$size" -ge $((20 * 1024 * 1024 * 1024)) ] || refuse "$disk holds $((size >> 30)) GB, and a box needs 20"
[ "$(id -u)" = 0 ] || refuse "writing a disk needs root: run it by sudo"

echo "$disk: $(lsblk -dno MODEL "$disk" | xargs), $((size >> 30)) GB. Everything on it is erased."
printf "Type %s to go on: " "$(basename "$disk")"
read -r answer
[ "$answer" = "$(basename "$disk")" ] || refuse "not confirmed"

dd if="$image" of="$disk" bs=4M conv=fsync oflag=direct status=progress
udevadm settle
if [ -n "$registration" ]; then  # onto the boot partition, the image's first
    boot=$(lsblk -lnpo NAME,TYPE "$disk" | awk '$2 == "part" { print $1; exit }')
    mounted=$(mktemp -d)
    mount "$boot" "$mounted"
    cp "$registration" "$mounted/leat-box.json"
    umount "$mounted"
    rmdir "$mounted"
    shred -u "$registration"
    echo "the registration is on the stick, and deleted here"
fi
sync
echo "done: boot the box of $disk"
