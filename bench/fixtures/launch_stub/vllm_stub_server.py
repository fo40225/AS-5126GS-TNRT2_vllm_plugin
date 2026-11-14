# Minimal /health and /metrics server for the vllm stub of bench/test_launch_flow.sh (the file name carries "vllm": the FB check looks for it in /proc/<pid>/cmdline).
import http.server, os, re, subprocess, sys, time

port = int(os.environ["STUB_PORT"])
delay = float(os.environ.get("STUB_READY_DELAY", "0"))
mode = os.environ.get("STUB_MODE", "ok")
t0 = time.time()
cci = open(os.environ["STUB_CCI_FILE"]).read()
sed = os.environ.get("STUB_CCI_SED")
if sed:
    cci = subprocess.run(["sed", "-E", sed], input=cci, capture_output=True, text=True).stdout

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/health"):
            ok = mode != "hang" and time.time() - t0 >= delay
            self.send_response(200 if ok else 503); self.end_headers()
        elif self.path.startswith("/metrics"):
            body = cci.encode(); self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a):
        pass

http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
