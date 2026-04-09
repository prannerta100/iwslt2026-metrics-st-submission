"""
SSL fix for pyenv Python 3.13 on macOS.
Import this FIRST before any other imports that make HTTPS requests.

This patches the default SSL context to use the macOS system keychain certs.
"""

import ssl
import os

# Generate cert bundle from macOS system keychain if not already done
cert_path = "/tmp/ca-bundle.crt"
if not os.path.exists(cert_path):
    os.system(
        "security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain > /tmp/ca-bundle.crt 2>/dev/null && "
        "security find-certificate -a -p /Library/Keychains/System.keychain >> /tmp/ca-bundle.crt 2>/dev/null"
    )

if os.path.exists(cert_path):
    os.environ["SSL_CERT_FILE"] = cert_path
    os.environ["REQUESTS_CA_BUNDLE"] = cert_path
    os.environ["CURL_CA_BUNDLE"] = cert_path

    # Monkey-patch ssl.create_default_context to always use our certs
    _original_create_default_context = ssl.create_default_context

    def _patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None, cadata=None):
        ctx = _original_create_default_context(purpose, cafile=cafile, capath=capath, cadata=cadata)
        if purpose == ssl.Purpose.SERVER_AUTH and cafile is None and capath is None and cadata is None:
            ctx.load_verify_locations(cafile=cert_path)
        return ctx

    ssl.create_default_context = _patched_create_default_context
    print(f"[ssl_fix] SSL cert bundle loaded from {cert_path}")
