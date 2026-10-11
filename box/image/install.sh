#!/bin/bash
# Installs the box's image onto this machine's internal disk, from a live system booted on it, as the
# Strix Halo or a laptop under test is made a box: the image streamed from serve.sh, which writes
# its address in for @URL@, compressed by zstd, and read back once written, its SHA-256 the image's,
# @SHA256@, of @SIZE@ bytes. The disk is an internal one alone, never a USB or removable one, and
# erased only once its name is typed. A boot entry, Leat, boots it from then on.
#
#     curl -s http://ADDRESS:8099/install | bash             # on the machine, as root
#     curl -s http://ADDRESS:8099/install | bash -s nvme0n1  # of one disk of several
set -euo pipefail
url=@URL@
size=@SIZE@
sha256=@SHA256@
refuse() { echo "refused: $1" >&2; exit 1; }

[ "$(id -u)" = 0 ] || refuse "installing needs root"
for tool in curl zstd dd wipefs lsblk; do
    command -v "$tool" >/dev/null || refuse "this live system has no $tool"
done
# the internal disks: whole ones, neither USB nor removable, as the live system's stick is, nor in
# memory, as zram's swap
internal=$(lsblk -dnpo NAME,TYPE,RM,TRAN \
    | awk '$2 == "disk" && $3 == 0 && $4 != "usb" && $1 !~ /^\/dev\/(zram|ram|loop)/ { print $1 }')
if [ -n "${1:-}" ]; then
    disk=/dev/${1#/dev/}
    echo "$internal" | grep -qx "$disk" || refuse "$disk is not an internal disk: $(echo $internal)"
elif [ "$(echo "$internal" | grep -c .)" = 1 ]; then
    disk=$internal
else
    refuse "name the disk of these to install onto: $(echo $internal)"
fi
room=$(lsblk -bdno SIZE "$disk")
[ "$room" -ge $((32 * 1024 * 1024 * 1024)) ] || refuse "$disk holds $((room >> 30)) GB, and a box needs 32"

echo "$disk: $(lsblk -dno MODEL "$disk" | xargs), $((room >> 30)) GB. Everything on it is erased."
printf "Type %s to go on: " "$(basename "$disk")"
read -r answer < /dev/tty
[ "$answer" = "$(basename "$disk")" ] || refuse "not confirmed"

wipefs -qa "$disk"
curl -sSf "$url/leat-box.raw.zst" | zstd -dc | dd of="$disk" bs=4M oflag=direct status=progress
sync
echo "checking what was written..."
written=$(head -c "$size" "$disk" | sha256sum | cut -d' ' -f1)
[ "$written" = "$sha256" ] || refuse "the disk does not hold the image: written again, it may; else the network, this machine's memory or its disk is at fault"
# a boot entry of its own, first in the firmware's order, as the old system's would point nowhere
if command -v efibootmgr >/dev/null && [ -d /sys/firmware/efi ]; then
    efibootmgr -q --create --disk "$disk" --part 1 --label Leat --loader '\EFI\BOOT\BOOTX64.EFI'
fi
echo "done: take the live system's stick out, and reboot"
