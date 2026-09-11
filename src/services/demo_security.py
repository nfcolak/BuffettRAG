"""Security primitives for the optional Streamlit demo."""
from __future__ import annotations

import html
import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Iterable, Mapping
from urllib.parse import urlparse


def _is_global_host(hostname: str, port: int) -> bool:
    """Resolve once during configuration; reject numeric and DNS private targets."""
    host = hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        address = ipaddress.ip_address(host)
        return address.is_global
    except ValueError:
        # Numeric IPv4 forms such as 2130706433 bypass ipaddress's dotted parser.
        if host.isdecimal():
            try:
                return ipaddress.ip_address(int(host, 10)).is_global
            except ValueError:
                return False
    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, ValueError):
        return False
    return bool(resolved) and all(ipaddress.ip_address(item[4][0]).is_global for item in resolved)


def validate_demo_backend_url(value: str, *, allowed_hosts: Iterable[str] = ()) -> str | None:
    """Accept only an allowlisted HTTPS host resolving exclusively to global IPs."""
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return None
    try:
        port = parsed.port or 443
    except ValueError:
        return None
    if port != 443:
        return None
    host = parsed.hostname.rstrip(".").lower()
    allowed = {str(item).rstrip(".").lower() for item in allowed_hosts}
    if not allowed or host not in allowed or not _is_global_host(host, port):
        return None
    return f"https://{host}"


def estimate_token_reservation(
    query: str, *, max_output_tokens: int, context_token_allowance: int
) -> int:
    """Reserve a conservative request budget before provider usage is known."""
    query_tokens = max(1, (len(query) + 2) // 3)
    return query_tokens + max(0, int(context_token_allowance)) + max(0, int(max_output_tokens))


def render_evidence_html(hit: Mapping[str, object], index: int) -> str:
    """Render backend evidence as text, never executable HTML."""
    safe_index = max(1, int(index))
    year = html.escape(str(hit.get("year", "?")), quote=True)
    source = html.escape(str(hit.get("source_file", "")), quote=True)
    text = html.escape(str(hit.get("text", "")), quote=True)
    return f'<div class="evidence-card"><b>[{safe_index}] · {year}</b> · {source}<br>{text}</div>'


@dataclass
class DemoRequestBudget:
    max_requests: int = 12
    max_tokens: int = 6000
    _usage: dict[str, tuple[int, int]] = field(default_factory=dict)

    def allow(self, session_id: str, *, tokens: int) -> bool:
        requests, used_tokens = self._usage.get(session_id, (0, 0))
        if requests >= self.max_requests or tokens < 1 or used_tokens + tokens > self.max_tokens:
            return False
        self._usage[session_id] = (requests + 1, used_tokens + tokens)
        return True
