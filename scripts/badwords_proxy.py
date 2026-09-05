#!/usr/bin/env python3
"""Reverse proxy that injects `bad_words` into every completion request.

For the decode-path token leak documented in docs/DECODE_PATH_TOKEN_LEAK.md: one vocabulary
token (`)Skip`, id 83480 in the DeepSeek-V4 tokenizer) is occasionally emitted at clause ends
because the small-batch decode/verify pass hands the sampler a corrupted logits row. Masking
the token makes the row fall back to the `.` or `,` that belongs there.

If you can edit your client, you do not need this -- just send
`"bad_words": [")Skip", ",Skip", ".Skip"]` with each request. This exists for clients that
cannot be changed. It adds no dependencies (standard library only) and streams responses
through untouched, so server-sent events still arrive token by token.

    python3 scripts/badwords_proxy.py --upstream http://127.0.0.1:30021 --listen 8000
    # then point the client at http://127.0.0.1:8000/v1

Only POSTs to paths ending in /chat/completions or /completions are rewritten; every other
request (including /tokenize, /v1/models and the health endpoints) passes through verbatim.

Not a scheduler and not hardened for the open internet: bind it to localhost.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_BAD_WORDS = [")Skip", ",Skip", ".Skip"]
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
       "trailers", "transfer-encoding", "upgrade", "content-length", "host"}

UPSTREAM = "http://127.0.0.1:30021"
BAD_WORDS = list(DEFAULT_BAD_WORDS)
VERBOSE = False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "badwords-proxy"

    def log_message(self, fmt, *args):
        if VERBOSE:
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _relay(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        path = self.path.split("?", 1)[0]

        if (method == "POST" and body
                and path.endswith(("/chat/completions", "/completions"))):
            body = self._inject(body)

        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        req = urllib.request.Request(UPSTREAM.rstrip("/") + self.path, data=body or None,
                                     headers=headers, method=method)
        try:
            up = urllib.request.urlopen(req, timeout=None)
        except urllib.error.HTTPError as e:
            up = e                      # relay the upstream error verbatim, status included
        except urllib.error.URLError as e:
            self.send_error(502, "upstream unreachable: %s" % e.reason)
            return

        try:
            self.send_response(up.status)
            for k, v in up.headers.items():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:                 # stream: SSE must not be buffered
                chunk = up.read(8192)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                        # client hung up mid-stream
        finally:
            up.close()

    def _inject(self, body):
        try:
            j = json.loads(body)
        except Exception:
            return body                 # not JSON: leave it alone
        if not isinstance(j, dict):
            return body
        have = j.get("bad_words")
        have = list(have) if isinstance(have, (list, tuple)) else []
        add = [w for w in BAD_WORDS if w not in have]
        if not add:
            return body
        j["bad_words"] = have + add
        return json.dumps(j).encode()

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")

    def do_DELETE(self):
        self._relay("DELETE")


def main():
    global UPSTREAM, BAD_WORDS, VERBOSE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", default=UPSTREAM, help="engine base URL")
    ap.add_argument("--listen", type=int, default=8000, help="port to listen on")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--bad-word", action="append", metavar="STR",
                    help="repeatable; replaces the default list %s" % DEFAULT_BAD_WORDS)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    UPSTREAM = a.upstream.rstrip("/")
    BAD_WORDS = a.bad_word if a.bad_word else list(DEFAULT_BAD_WORDS)
    VERBOSE = a.verbose
    srv = ThreadingHTTPServer((a.host, a.listen), Handler)
    srv.daemon_threads = True
    print("badwords-proxy :%d -> %s . injecting %s" % (a.listen, UPSTREAM, BAD_WORDS), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
