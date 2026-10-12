# The box

What makes a machine running leat an office's Leat, reached at `https://<id>.leatbox.net` with a
certificate every browser trusts, as no certificate can be of `leat.local`. leatbox.com's server,
of the `leatbox-server` repository, keeps the boxes' names and answers their certificates'
challenges.

```
browser ── https://k7x2qm.leatbox.net ──▶ Caddy (leat-proxy) ──▶ leat agent on 127.0.0.1:8000
                                            │ certificate: DNS-01, through the box's acme-dns account
                                            ▼
                               https://api.leatbox.net ◀── leat.box.address: the name's address
```

## Its parts

| | |
|---|---|
| `build-caddy.sh` | builds `leat-caddy`, Caddy of a pinned release with the acme-dns module, which goes to `/usr/local/bin/leat-caddy` |
| `Caddyfile` | the front door, at `/usr/share/leat/box/Caddyfile`: the app at the box's name over HTTPS, HTTP/2 and 3; plain HTTP, as `http://leat.local`, redirected there |
| `systemd/leat-proxy.service` | runs it, as a user of its own, on 80 and 443 |
| `systemd/leat-address.service`, `.timer` | `python -m leat.box.address`: at boot and every 30 s, the registry is told the box's address on the office's network, when it changed or a day went by |
| `leat/box/provision.py` | writes a box's registration into `/etc/leat`, when we build it |

leat agent runs with `--name <id>.leatbox.net`, on loopback alone, behind Caddy.

## Making a box

```bash
box/build-caddy.sh leat-caddy                                    # needs Go
ssh leatbox leatbox-registry register | sudo python -m leat.box.provision
```

Provisioning writes `/etc/leat/box.env` (its ID, name and the registry's address), and, root's
alone, `registry-secret` and `acme-dns.json`, which systemd gives the services as credentials.
On its first boot the box tells the registry its address and Caddy has its certificate, in some
ten seconds.

## What the box reaches on the internet, for this

- `https://api.leatbox.net`: its name's address, and its certificate's challenges.
- Let's Encrypt: its certificate, every two months or so.
- Public DNS resolvers (1.1.1.1, 9.9.9.9, 8.8.8.8) on port 53, to check its challenges: an
  office's router may refuse those lookups, as one in Turkey answered its own data as REFUSED.

None of it carries anything of the office's work. Its name and address, as every certificate's
name, are public: a name points at a private address, reachable inside its office alone.

## Set up without a cable

A box that has gone a minute without a network, or five if it knows a Wi-Fi network that may come
back, opens a hotspot of its own (`leat-hotspot.service`, `python -m leat.box.hotspot`): its radio
becomes iwd's access point, `Leat-` and four of its ID's letters, with a passphrase the screen
shows beside a QR code that joins it. systemd-networkd gives the hotspot `10.42.0.1` and its devices
addresses by DHCP (`82-hotspot.network`). Every name a phone on it asks for is the box's, as the
hotspot answers its DNS, and Caddy sends every request made to the hotspot's address to its page,
which the phone shows as a captive portal: the networks in reach and a passphrase. A network chosen, the
hotspot closes and the box joins it. One it can't join in half a minute is forgotten and the hotspot
opens again, saying so. It closes, too, once the box has a network of its own: a cable, or a known
network back in reach, as it scans every minute.

Tried in a VM, of `modprobe mac80211_hwsim radios=3`: the box's radio, a phone's and an office's
access point (iwd's `DriverQuirks` `DefaultInterface=*` there, as hwsim's radios refused iwd's own
interfaces).

## Its disk

`box/image` builds it with mkosi. The image holds the first three partitions; the box's first boot
makes the last two, of the rest of its disk.

| Partition | Size | What |
|---|---|---|
| ESP | 512 MB | systemd-boot, and a UKI of each system, `leat_<version>.efi` |
| `leat_<version>` | 8 GB | the system, A, of the image, replaced whole by an update |
| `_empty` | 8 GB | the system, B, which an update writes, the two taking turns |
| `leat-identity` | 64 MB | `/etc/leat`: the box's name, registration, and its certificate and account, Caddy's |
| `leat-data` | the rest | `/var`: the office's data, and the system's own state, Wi-Fi, certificates, journal |

The last two are encrypted, sealed to the TPM by a signed PCR policy: they unlock for a UKI of
ours alone, of any version, and a disk taken out of a box gives nothing away. The identity outlives
updates and a factory reset; the data outlives updates. A factory reset, an admin's, puts a flag on
the ESP, and the boot after deletes the data's partition and makes it anew, of a new key.

## Updates

Every few hours the box looks at `https://leatbox.com/updates/` (systemd-sysupdate, of
`/usr/lib/sysupdate.d`), checks its `SHA256SUMS` against our key, and writes a newer system into
its other partition and its UKI beside the running one, with three tries. At night it restarts into
it (`systemd-sysupdate-reboot.timer`); the boot is kept once Leat answers (`leat-health.service`,
before `boot-complete.target`), and else, its tries spent, systemd-boot boots the system before. An
admin sees the version, and an update ready, in Settings.

```bash
box/image/keys.sh                     # once: the keys, kept in ~/.config/leat/box-keys
cd box/image && mkosi build           # a product build, of a commit, without the dev profile
./publish.sh --upload                 # onto leatbox.com/updates, for every box to take
```

**Back up `~/.config/leat/box-keys`.** Its PCR key signs every UKI the boxes' TPMs unlock their
data for, and its GPG key every update they take: lost, no box takes an update again.

## Trying it on a machine of your own

On ports 8443 and 8088, without root, its state in a folder of its own:

```bash
D=~/.local/share/leat-devbox
ssh leatbox leatbox-registry register | uv run python -m leat.box.provision --root $D/etc
uv run python -m leat.box.address --root $D/etc --state $D/address
N=$(sed -n 's/^LEAT_NAME=//p' $D/etc/box.env)
uv run leat agent --name $N --data $D/data &
LEAT_NAME=$N CREDENTIALS_DIRECTORY=$D/etc LEAT_HTTPS_PORT=8443 LEAT_HTTP_PORT=8088 \
    XDG_DATA_HOME=$D/caddy XDG_CONFIG_HOME=$D/caddy \
    leat-caddy run --config box/Caddyfile --adapter caddyfile
```

`LEAT_ACME_CA=https://acme-staging-v02.api.letsencrypt.org/directory` has its certificate of Let's
Encrypt's staging CA, which no browser trusts, as tests should.
