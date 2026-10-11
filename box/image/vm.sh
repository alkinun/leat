#!/bin/sh
# Boots the image just built, as a fresh box would, in a VM of its own: a copy of the image on a
# 24 GB disk, which its first boot partitions; its console logged to ~/.cache/leat-vm/serial.log;
# SSH, HTTPS and HTTP forwarded to this machine's 2222, 9443 and 9080. Stops the VM it booted before.
#
#     box/image/vm.sh            # then: ssh leat-vm (of ~/.ssh/config), or https://127.0.0.1:9443
#     box/image/vm.sh stop
set -eu
image=$(dirname "$(realpath "$0")")/mkosi.output/leat-box.raw
vm=$HOME/.cache/leat-vm
firmware=/usr/share/edk2/x64
mkdir -p "$vm"
if [ -f "$vm/qemu.pid" ]; then
    kill "$(cat "$vm/qemu.pid")" 2>/dev/null || true
    rm -f "$vm/qemu.pid"
    sleep 1
fi
[ "${1:-}" = stop ] && exit 0
rm -f "$vm/disk.raw" "$vm/serial.log"
cp --sparse=always "$image" "$vm/disk.raw"
truncate -s 24G "$vm/disk.raw"
cp "$firmware/OVMF_VARS.4m.fd" "$vm/vars.fd"
qemu-system-x86_64 -enable-kvm -cpu host -smp 4 -m 4G -machine q35 \
    -drive if=pflash,format=raw,readonly=on,file="$firmware/OVMF_CODE.4m.fd" \
    -drive if=pflash,format=raw,file="$vm/vars.fd" \
    -drive file="$vm/disk.raw",format=raw,if=virtio \
    -nic user,model=virtio-net-pci,hostfwd=tcp:127.0.0.1:2222-:22,hostfwd=tcp:127.0.0.1:9443-:443,hostfwd=tcp:127.0.0.1:9080-:80 \
    -display none -serial file:"$vm/serial.log" -monitor none -daemonize -pidfile "$vm/qemu.pid"
for _ in $(seq 60); do
    grep -q "Reached target.*Multi-User System" "$vm/serial.log" 2>/dev/null && break
    sleep 1
done
grep -o "Reached target.*Multi-User System" "$vm/serial.log" >/dev/null && echo "booted: ssh leat-vm" \
    || { echo "not booted in 60 s: see $vm/serial.log"; exit 1; }
