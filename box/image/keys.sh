#!/bin/sh
# Makes the box's signing keys, once, and links them where mkosi takes them: the key that signs the
# PCR values each image's UKI expects, which the TPM unlocks the box's encrypted partitions by, any
# image of ours alike; and the GPG key that signs its updates. Their public halves are committed,
# into the image; the private ones are kept in ~/.config/leat/box-keys alone, and must be backed up:
# lost, no box ever takes an update again, nor unlocks its data under a new one.
#
#     box/image/keys.sh
set -eu
here=$(dirname "$(realpath "$0")")
keys=${LEAT_BOX_KEYS:-$HOME/.config/leat/box-keys}
mkdir -p "$keys" && chmod 700 "$keys"
if [ ! -f "$keys/pcr.key" ]; then
    openssl req -quiet -new -x509 -newkey rsa:2048 -nodes -days 36500 -subj "/CN=Leat box" \
        -keyout "$keys/pcr.key" -out "$keys/pcr.crt"
    chmod 600 "$keys/pcr.key"
fi
if [ ! -d "$keys/gnupg" ]; then
    mkdir -m 700 "$keys/gnupg"
    gpg --homedir "$keys/gnupg" --batch --passphrase '' \
        --quick-gen-key "Leat box updates <updates@leatbox.com>" ed25519 sign never 2>/dev/null
fi
ln -sf "$keys/pcr.key" "$here/mkosi.key"
ln -sf "$keys/pcr.crt" "$here/mkosi.crt"
extra=$here/mkosi.extra/usr/lib/systemd
mkdir -p "$extra"
openssl x509 -in "$keys/pcr.crt" -pubkey -noout > "$extra/tpm2-pcr-public-key.pem"
gpg --homedir "$keys/gnupg" --export > "$extra/import-pubring.pgp"
echo "the box's keys: $keys (back them up)"
