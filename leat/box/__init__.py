"""The box: what makes a machine running leat an office's Leat, reached at its own name.

Each box is registered with leatbox.com's registry when we build it, and has a name, as
k7x2qm.leatbox.net, which points at its address on the office's network. Caddy, in front of
leat agent, serves the app at that name over HTTPS, with a certificate it has from Let's Encrypt by
DNS-01, through the box's acme-dns account; leat.box.address keeps the name pointing at the box as
its address changes. Nothing of the office's work leaves the box for either.

The box keeps what registration gave it in /etc/leat, as leat.box.provision writes it.
"""
