# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub:
[open a security advisory](https://github.com/centuryglass/sd-backend-client/security/advisories/new) (the
"Report a vulnerability" button on the repository's Security tab). Don't open a public issue, pull request or
discussion for one.

Include what an attacker can do, the affected version or commit, and the steps or a minimal snippet that reproduce it.
This is a one-maintainer project, so there is no fixed response time; reports are acknowledged as soon as possible,
and fixed issues are credited in the advisory unless you ask otherwise.

## Supported versions

Fixes land on `main` and ship in the next release. Older releases don't get backported fixes.

## Scope

In scope is the code in this repository, in particular:

- handling of WebUI credentials: the `credentials_provider` callback and the HTTP basic auth it configures.
- how the client builds requests and parses responses from a server it is pointed at, including a malicious or
  compromised one.

Vulnerabilities in ComfyUI, Stable Diffusion WebUI (Automatic1111, Forge, reForge, Forge Neo) or their extensions belong with
those projects.

## Using the client safely

- The client sends WebUI credentials as HTTP basic auth, which is readable on the network unless the server URL uses
  `https://`. Use TLS, or keep the connection on a trusted host or network.
- ComfyUI has no authentication. Anyone who can reach a ComfyUI server can queue work on it, so don't expose one
  beyond a trusted network.
