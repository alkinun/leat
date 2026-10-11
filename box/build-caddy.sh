#!/bin/sh
# Builds the box's Caddy: a pinned release, with the acme-dns DNS module by which it has its
# certificate. Go checks every module against its checksum database. Needs Go.
#
#     box/build-caddy.sh [output]    # ./leat-caddy by default
set -eu
caddy=v2.11.7
xcaddy=v0.4.7
acmedns=v0.7.0
out=$(realpath -m "${1:-leat-caddy}")
tools=$(mktemp -d)
trap 'rm -rf "$tools"' EXIT
GOBIN="$tools" go install "github.com/caddyserver/xcaddy/cmd/xcaddy@$xcaddy"
"$tools/xcaddy" build "$caddy" --with "github.com/caddy-dns/acmedns@$acmedns" --output "$out"
"$out" list-modules | grep -qx dns.providers.acmedns
"$out" version
