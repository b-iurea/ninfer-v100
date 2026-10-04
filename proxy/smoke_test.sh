#!/usr/bin/env bash
# End-to-end check of ninfer-proxy against stub ninfer-serve / ninfer-perplexity (no GPU needed).
# usage: proxy/smoke_test.sh [path/to/ninfer-proxy]
set -eu  # no pipefail: grep -q closes the pipe early and curl would report SIGPIPE
BIN=$(realpath "${1:-$(dirname "$0")/ninfer-proxy}")
T=$(mktemp -d); trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$T"' EXIT
cd "$T"

cat > fake-serve <<'EOF'
#!/usr/bin/env python3
import sys, json, time, http.server
a = sys.argv; port = int(a[a.index("--port") + 1]); mid = a[a.index("--model-id") + 1]
print("loading", a[1], flush=True); time.sleep(1); print("warn: fake", flush=True)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def send(self, body, ctype="application/json"):
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body.encode())
    def do_GET(self):
        if self.path == "/metrics": self.send("# HELP x\nninfer_last_tokens_per_second 42.5\nninfer_running_requests 0\n", "text/plain")
        else: self.send('{"status":"ok"}')
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send("data: " + json.dumps({"served_by": mid, "model": req["model"]}) + "\n\ndata: [DONE]\n\n", "text/event-stream")
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
EOF
cat > fake-ppl <<'EOF'
#!/usr/bin/env python3
import sys, os, json
out = sys.argv[sys.argv.index("--output") + 1]; os.makedirs(out)
print("scoring | PPL 6.3", flush=True)
json.dump({"overall": {"perplexity": 6.3631}, "domains": [{"domain": "en", "perplexity": 5.1}]}, open(out + "/report.json", "w"))
EOF
chmod +x fake-serve fake-ppl
touch a.ninfer b.ninfer
cat > config.yaml <<EOF
listen: "127.0.0.1:18080"
serve_bin: $T/fake-serve
perplexity_bin: $T/fake-ppl
corpus: /dev/null
port_start: 19100
models:
  a: {path: $T/a.ninfer, args: ["--kv-dtype", "int8"]}
  b: {path: $T/b.ninfer, port: 19200}
EOF
"$BIN" -config config.yaml > proxy.log 2>&1 &
P=http://127.0.0.1:18080
for _ in $(seq 50); do curl -sf $P/api/models >/dev/null && break; sleep 0.1; done
fail() { echo "FAIL $1"; cat proxy.log; exit 1; }
state() { curl -s $P/api/models | python3 -c "import sys,json;print({m['id']:m['state'] for m in json.load(sys.stdin)}['$1'])"; }

curl -sf $P/ | grep -q "Ninfer V100 Turbo" || fail "UI served"
curl -s $P/v1/models | grep -q '"id":"b"' || fail "/v1/models lists config"

[ "$(curl -s -o /dev/null -w '%{http_code}' $P/health)" = 503 ] || fail "/health 503 with no model loaded"

# on-demand load + streaming proxy routed by "model"
r=$(curl -sN $P/v1/chat/completions -d '{"model":"a","stream":true}')
grep -q '"served_by": "a"' <<<"$r" || fail "request routed to a (auto-load)"
[ "$(state a)" = ready ] || fail "a ready"
curl -s "$P/api/models/a/metrics" | grep -q 42.5 || fail "model metrics"
curl -sf $P/health | grep -q '"model":"a"' || fail "/health ready with a"
curl -sf $P/metrics | grep -q "ninfer_last_tokens_per_second 42.5" || fail "/metrics passes through"
curl -s "$P/api/logs?model=a&since=0" | grep -q -- "--kv-dtype int8" || fail "logs captured with args"

# exclusive: loading b unloads a, b uses its fixed port
r=$(curl -sN $P/v1/chat/completions -d '{"model":"b"}')
grep -q '"served_by": "b"' <<<"$r" || fail "routed to b"
[ "$(state a)" = stopped ] || fail "exclusive swap stopped a"
curl -s $P/api/models | grep -q '"port":19200' || fail "fixed port honored"
curl -s -o /dev/null -w '%{http_code}' $P/v1/chat/completions -d '{"model":"zzz"}' | grep -q 404 || fail "unknown model 404"

# perplexity job (unloads b)
curl -sf $P/api/ppl -d '{"model":"a","mode":"quick","kv":"fp8"}' >/dev/null
for _ in $(seq 50); do curl -s $P/api/ppl | grep -q '"state":"done"' && break; sleep 0.1; done
curl -s $P/api/ppl | grep -q '"ppl":6.3631' || fail "perplexity result parsed"
[ "$(state b)" = stopped ] || fail "ppl freed the GPU"

# config: invalid rejected, valid saved and applied
code=$(curl -s -o /dev/null -w '%{http_code}' -X PUT $P/api/config --data-binary 'models: {x: {port: 1}}')
[ "$code" = 400 ] || fail "invalid config rejected"
sed 's/^  b: .*$//' config.yaml > new.yaml
curl -sf -X PUT $P/api/config --data-binary @new.yaml >/dev/null
! curl -s $P/api/models | grep -q '"id":"b"' || fail "config saved, b removed"
curl -s $P/api/stats | grep -q '"mem_total"' || fail "host stats"

# settings: patch keeps comments and models; api_key then guards /v1 and /api, not the UI page
echo "load_timeout: 900 # keep me" >> config.yaml
curl -sf -X PUT $P/api/settings -d '{"load_timeout":120,"cors":true,"api_key":"s3cret"}' >/dev/null || fail "settings saved"
grep -q "load_timeout: 120 # keep me" config.yaml || fail "settings kept the comment"
grep -q "a: {path:" config.yaml || fail "settings kept models"
[ "$(curl -s -o /dev/null -w '%{http_code}' $P/api/models)" = 401 ] || fail "console needs key"
[ "$(curl -s -o /dev/null -w '%{http_code}' $P/v1/models)" = 401 ] || fail "/v1 needs key"
[ "$(curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer nope' $P/v1/models)" = 401 ] || fail "wrong key rejected"
curl -sf -H 'Authorization: Bearer s3cret' $P/v1/models | grep -q '"id":"a"' || fail "bearer accepted"
curl -sf -H 'x-api-key: s3cret' $P/api/settings | grep -q '"load_timeout":120' || fail "x-api-key accepted"
curl -sf $P/ >/dev/null || fail "UI page stays public"
[ "$(curl -s -o /dev/null -w '%{http_code}' $P/metrics)" = 401 ] || fail "/metrics needs key"
[ "$(curl -s -o /dev/null -w '%{http_code}' $P/health)" != 401 ] || fail "/health stays open"
curl -s -D - -o /dev/null -X OPTIONS $P/v1/chat/completions | grep -qi "access-control-allow-origin: \*" || fail "CORS preflight"
code=$(curl -s -o /dev/null -w '%{http_code}' -H 'x-api-key: s3cret' -X PUT $P/api/settings -d '{"port_start":0}')
[ "$code" = 400 ] || fail "invalid setting rejected"
echo "all checks passed"
