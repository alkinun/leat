#!/bin/sh
# Boots the image just built, as a fresh box would, in a VM of its own: a copy of the image on a
# 24 GB disk, which its first boot partitions, and a TPM of swtpm's, which its encrypted partitions
# are sealed to; its serial port logged to ~/.cache/leat-vm/serial.log;
# SSH, HTTPS and HTTP forwarded to this machine's 2222, 9443 and 9080; its QEMU monitor, as for
# screenshots of its display, at monitor.sock. Stops the VM it booted before.
#
#     box/image/vm.sh                  # then: ssh leat-vm (of ~/.ssh/config)
#     box/image/vm.sh leat-box.json    # with a box's registration on its boot partition, as a
#                                      # stick's is put: leatbox-registry register's JSON
#     box/image/vm.sh again            # the VM cut off, as a power cut does, and booted again, of
#                                      # the same disk and TPM
#     box/image/vm.sh stop
#
# A USB disk, an image of a file system, as mkfs.fat makes, is plugged into its USB port as into a
# box's, of the QEMU monitor's commands, and unplugged of "device_del stick":
#
#     drive_add 0 if=none,id=stick,format=raw,file=/path/to/stick.raw
#     device_add usb-storage,bus=usb.0,drive=stick,id=stick
#
# Of 2 GB of memory, or $LEAT_VM_MEMORY; its disk written past this machine's page cache, as an
# update's 8 GB written into it would fill its memory else.
set -eu
here=$(dirname "$(realpath "$0")")
image=$here/mkosi.output/leat-box.raw
vm=$HOME/.cache/leat-vm
firmware=/usr/share/edk2/x64
mkdir -p "$vm"
if [ -f "$vm/qemu.pid" ]; then
    kill -KILL "$(cat "$vm/qemu.pid")" 2>/dev/null || true
    rm -f "$vm/qemu.pid"
    sleep 1
fi
if [ -f "$vm/swtpm.pid" ]; then
    kill "$(cat "$vm/swtpm.pid")" 2>/dev/null || true
    rm -f "$vm/swtpm.pid"
fi
[ "${1:-}" = stop ] && exit 0
rm -f "$vm/serial.log"
if [ "${1:-}" != again ]; then
    rm -rf "$vm/disk.raw" "$vm/tpm"
    cp --sparse=always "$image" "$vm/disk.raw"
    truncate -s 24G "$vm/disk.raw"
    cp "$firmware/OVMF_VARS.4m.fd" "$vm/vars.fd"
fi
if [ -n "${1:-}" ] && [ "$1" != again ]; then  # into the boot partition, the image's first, a FAT one
    start=$(sfdisk -J "$vm/disk.raw" | python3 -c "import json, sys; print(json.load(sys.stdin)['partitiontable']['partitions'][0]['start'] * 512)")
    (cd "$here" && mkosi sandbox -- mcopy -o -i "$vm/disk.raw@@$start" "$(realpath "$1")" ::leat-box.json)
fi
mkdir -p "$vm/tpm"
(cd "$here" && mkosi sandbox -- swtpm socket --tpm2 --tpmstate dir="$vm/tpm" \
    --ctrl type=unixio,path="$vm/swtpm.sock" --pid file="$vm/swtpm.pid" --daemon)
qemu-system-x86_64 -enable-kvm -cpu host -smp 4 -m "${LEAT_VM_MEMORY:-2G}" -machine q35 \
    -chardev socket,id=tpm,path="$vm/swtpm.sock" -tpmdev emulator,id=tpm,chardev=tpm \
    -device tpm-tis,tpmdev=tpm -device qemu-xhci,id=usb \
    -drive if=pflash,format=raw,readonly=on,file="$firmware/OVMF_CODE.4m.fd" \
    -drive if=pflash,format=raw,file="$vm/vars.fd" \
    -drive file="$vm/disk.raw",format=raw,if=virtio,cache=none \
    -nic user,model=virtio-net-pci,hostfwd=tcp:127.0.0.1:2222-:22,hostfwd=tcp:127.0.0.1:9443-:443,hostfwd=tcp:127.0.0.1:9080-:80 \
    -display none -serial file:"$vm/serial.log" -monitor unix:"$vm/monitor.sock",server,nowait \
    -daemonize -pidfile "$vm/qemu.pid"
# booted once it answers: by SSH, of a dev build, or its front door, of any provisioned one
for _ in $(seq 90); do
    if ssh -p 2222 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -o ConnectTimeout=1 root@127.0.0.1 true 2>/dev/null; then
        echo "booted: ssh leat-vm"
        exit 0
    fi
    if curl -s -o /dev/null --max-time 1 http://127.0.0.1:9080/; then
        echo "booted: no SSH, as a product build has none"
        exit 0
    fi
    sleep 1
done
echo "no answer in 90 s: see $vm/serial.log"
exit 1
