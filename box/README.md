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
| `Caddyfile` | the front door, at `/usr/share/leat/box/Caddyfile`: the app at the box's name over HTTPS, HTTP/2 and 3; plain HTTP, as `http://leat.local`, redirected there |
| `build-caddy.sh` | builds `leat-caddy`, Caddy of a pinned release with the acme-dns module; the image's build runs it |
| `systemd/leat-provision.service` | `python -m leat.box.provision`: on the first boot, takes in the registration a stick or VM put on the boot partition, into `/etc/leat` |
| `systemd/leat-enroll.service` | `python -m leat.box.enroll`: on the first boot of a box with no registration, registers it of its TPM, as its maker's fingerprint of it allows |
| `systemd/leat-proxy.service` | Caddy, as a user of its own, on 80 and 443 |
| `systemd/leat-agent.service` | leat agent, with `--name <id>.leatbox.net`, on loopback alone, behind Caddy |
| `systemd/leat-searxng.service` | SearXNG, the web's search, for Leat alone, on loopback |
| `systemd/leat-address.service`, `.timer` | `python -m leat.box.address`: at boot and every 30 s, the registry is told the box's address on the office's network, when it changed or a day went by |
| `systemd/leat-screen.service` | `python -m leat.box.screen`: what a display plugged into the box shows, and its power button |
| `systemd/leat-system.socket`, `leat-system@.service` | `python -m leat.box.system`: the few changes of the system leat agent asks for as root, each checked: time zone, fixed address, power, factory reset, updates, diagnostics |
| `systemd/leat-system-restore.service` | puts back, at boot, the time zone and fixed address an update's new system lacks |
| `systemd/leat-health.service`, `leat-fallback.service` | an update's trial boot kept once Leat answers, and restarted if it never does, or falls into emergency mode, until systemd-boot goes back to the system before |
| `/etc/leat/engine.env` | the engine leat agent uses: the image runs no `leat serve` and holds no model yet, so the agent reaches `http://127.0.0.1:8080`, or the engine this file names (`LEAT_ENGINE`, and `LEAT_ENGINE_KEY` if it asks for one), as a box under test reaches one on another machine, its prompts leaving the box |
| `image/` | the disk image, of mkosi: below |

`leat.box.watch`, in leat agent, says what needs attention, on the screen and in Settings: a disk
nearly full, the model or the internet out of reach, a certificate about to expire, the name not
found, an update that failed.

## Making a box

```bash
box/image/keys.sh                                       # once: the keys that sign the images
cd box/image && mkosi --profile dev -f build            # mkosi.output/leat-box.raw
ssh leatbox leatbox-registry register > leat-box.json   # a box's registration: its name, keys
```

Then one of:

- `box/image/vm.sh leat-box.json`: a VM of it, with a TPM, as below.
- `sudo box/image/flash.sh /dev/sdX leat-box.json`: a USB stick a machine boots as a box, the
  registration on its boot partition.
- `box/image/serve.sh`, then, on a machine booted on a live system,
  `curl -s http://ADDRESS:8099/install | bash`: the image installed onto its internal disk. It has
  no registration yet: put `leat-box.json` on its boot partition before it first boots, or it
  says "Not set up".

A box we ship registers itself instead, on its first boot, by its TPM (`leat-enroll`): its maker
runs `python -m leat.box.enroll --fingerprint` on each, or reads the endorsement key's
fingerprint as the SHA-256 of its public key's DER, and we allow each on the server:

```bash
ssh leatbox leatbox-registry allow FINGERPRINT --serial SERIAL
```

The box gives the registry its endorsement key and an attestation key's name, takes the secret of
the credential it is answered out of its TPM, which only that TPM can, and is answered its
registration. One whose disk was erased registers itself again of its own name.

The dev profile adds an SSH key, of the git-ignored `mkosi.local.conf`; a box we ship has none. On
its first boot, `leat-provision` writes `/etc/leat/box.env` (its ID, name and the registry's
address) and, root's alone, `registry-secret` and `acme-dns.json`, which systemd gives the
services as credentials. The box tells the registry its address and Caddy has its certificate, in
some ten seconds. `box/image/initrd.py` lists the initrd's files, as when a driver is missing.

## What the box reaches on the internet, for this

- `https://api.leatbox.net`: its name's address, and its certificate's challenges.
- Let's Encrypt: its certificate, every two months or so.
- Public DNS resolvers (1.1.1.1, 9.9.9.9, 8.8.8.8) on port 53, to check its challenges: an
  office's router may refuse those lookups, as one in Turkey answered its own data as REFUSED.

None of it carries anything of the office's work. Its name and address, as every certificate's
name, are public: a name points at a private address, reachable inside its office alone.

## A fixed address

An office whose IT gives the box its own address sets it in Settings, The box, Network: an address
and its prefix, a gateway and DNS servers, which the box's helper writes as
`/run/systemd/network/10-leat-fixed.network`, ahead of the wired link's DHCP. It is kept on the data
partition and written again at each boot. Forty seconds after one is set, a check asks whether the
gateway answers on the link, by its neighbour's entry. If it doesn't, as of an address mistyped, the
box goes back to DHCP by itself and Settings says so, so a box is never lost to a wrong one.

## Its disk

`box/image` builds it with mkosi. The image holds the first three partitions; the box's first boot
makes the last two, of the rest of its disk.

| Partition | Size | What |
|---|---|---|
| ESP | 512 MB | systemd-boot, and a UKI of each system, `leat_<version>.efi` |
| `leat_<version>` | 8 GB | the system, A, of the image, replaced whole by an update |
| `_empty` | 8 GB | the system, B, which an update writes, the two taking turns |
| `leat-identity` | 64 MB | `/etc/leat`: the box's name, registration, and its certificate and account, Caddy's |
| `leat-data` | the rest | `/var`: the office's data, and the system's own state, certificates, journal |

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
update cut off while it downloads, as by a power cut, is removed before the next is tried, as
systemd-sysupdate would refuse every update after it else. An admin sees the version, and an
update ready, in Settings.

Two channels: a box takes `https://leatbox.com/updates/`, of builds without the dev profile; a box
under test, of a dev image, takes `https://leatbox.com/updates/dev/`, of dev builds alone, as a
dev build's SSH must never reach a box, and a box under test must keep its own.
`mkosi.postinst.chroot` points a dev image at its channel and marks each image's in
`/usr/lib/leat/channel`, which `publish.sh` publishes it to.

A release, of a commit, everything committed and its tests passing:

```bash
box/image/keys.sh                     # once: the keys, kept in ~/.config/leat/box-keys
cd box/image
mkosi --profile dev -f build          # the dev image first, for the boxes under test
./publish.sh --upload                 # onto leatbox.com/updates/dev
mkosi -f build                        # then the boxes'
./publish.sh --upload                 # onto leatbox.com/updates
```

A box under test shows it took the update in Settings, and its version on its screen; a box takes
it within a few hours, and restarts into it at night.

**Back up `~/.config/leat/box-keys`.** Its PCR key signs every UKI the boxes' TPMs unlock their
data for, and its GPG key every update they take: lost, no box takes an update again.

## Trying it on a machine of your own

On ports 8443 and 8088, without root, its state in a folder of its own, of `box/build-caddy.sh
leat-caddy`'s Caddy, which needs Go:

```bash
D=~/.local/share/leat-devbox
ssh leatbox leatbox-registry register | uv run python -m leat.box.provision --root $D/etc
uv run python -m leat.box.address --root $D/etc --state $D/address
N=$(sed -n 's/^LEAT_NAME=//p' $D/etc/box.env)
uv run leat agent --name $N --data $D/data &
LEAT_NAME=$N CREDENTIALS_DIRECTORY=$D/etc LEAT_HTTPS_PORT=8443 LEAT_HTTP_PORT=8088 \
    XDG_DATA_HOME=$D/caddy XDG_CONFIG_HOME=$D/caddy \
    ./leat-caddy run --config box/Caddyfile --adapter caddyfile
```

`LEAT_ACME_CA=https://acme-staging-v02.api.letsencrypt.org/directory` has its certificate of Let's
Encrypt's staging CA, which no browser trusts, as tests should.
