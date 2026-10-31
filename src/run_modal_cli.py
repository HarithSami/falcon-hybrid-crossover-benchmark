"""
Run Modal CLI with automatic Cloudflare (1.1.1.1) DNS fallback if local ISP DNS fails
to resolve *.modal.com.
Usage:
  python src/run_modal_cli.py run src/modal_benchmark.py --mode smoke --context-lengths 512
"""
import socket
import subprocess
import sys

_orig_getaddrinfo = socket.getaddrinfo
_dns_cache: dict[str, str] = {}


def _resolve_via_1111(host: str) -> str | None:
    if host in _dns_cache:
        return _dns_cache[host]
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Resolve-DnsName -Name '{host}' -Type A -Server 1.1.1.1 | Where-Object {{ $_.IPAddress }} | Select-Object -First 1).IPAddress",
            ],
            text=True,
            timeout=5,
        ).strip()
        if out:
            _dns_cache[host] = out
            return out
    except Exception:
        pass
    return None


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _orig_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        if isinstance(host, str) and "modal" in host:
            ip = _resolve_via_1111(host)
            if ip:
                return _orig_getaddrinfo(ip, port, family, type, proto, flags)
        raise


socket.getaddrinfo = _patched_getaddrinfo

if __name__ == "__main__":
    from modal.__main__ import main

    sys.argv = ["modal"] + sys.argv[1:]
    main()
