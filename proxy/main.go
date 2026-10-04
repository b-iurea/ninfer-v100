// ninfer-proxy: one port (8080) in front of many ninfer-serve processes, plus a small web UI
// to load/unload models, tail their logs, watch host/GPU metrics, run perplexity and edit config.
package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
)

//go:embed index.html
var indexHTML []byte

type ModelCfg struct {
	Path        string   `yaml:"path" json:"path"`
	Port        int      `yaml:"port,omitempty" json:"port,omitempty"`
	Args        []string `yaml:"args,omitempty" json:"args,omitempty"`
	Description string   `yaml:"description,omitempty" json:"description,omitempty"`
}

type Config struct {
	Listen      string              `yaml:"listen"`
	ServeBin    string              `yaml:"serve_bin"`
	PplBin      string              `yaml:"perplexity_bin"`
	Corpus      string              `yaml:"corpus"`
	PortStart   int                 `yaml:"port_start"`
	Exclusive   bool                `yaml:"exclusive"`
	LoadTimeout int                 `yaml:"load_timeout"`
	Models      map[string]ModelCfg `yaml:"models"`
}

func parseConfig(data []byte) (*Config, error) {
	c := &Config{Listen: ":8080", ServeBin: "ninfer-serve", PplBin: "ninfer-perplexity",
		PortStart: 9100, Exclusive: true, LoadTimeout: 900}
	if err := yaml.Unmarshal(data, c); err != nil {
		return nil, err
	}
	used := map[int]string{}
	for id, m := range c.Models {
		if strings.TrimSpace(m.Path) == "" {
			return nil, fmt.Errorf("model %q: path is required", id)
		}
		if m.Port != 0 {
			if other, dup := used[m.Port]; dup {
				return nil, fmt.Errorf("models %q and %q share port %d", id, other, m.Port)
			}
			used[m.Port] = id
		}
	}
	return c, nil
}

// Log is a bounded line buffer; clients poll it with ?since=<seq>.
type Log struct {
	mu    sync.Mutex
	lines []string
	base  int // seq of lines[0]
	part  []byte
}

const maxLogLines = 5000

func (l *Log) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.part = append(l.part, p...)
	for {
		i := bytes.IndexAny(l.part, "\r\n")
		if i < 0 {
			break
		}
		if line := strings.TrimRight(string(l.part[:i]), " "); line != "" {
			l.lines = append(l.lines, line)
		}
		l.part = l.part[i+1:]
	}
	if n := len(l.lines) - maxLogLines; n > 0 {
		l.lines = append([]string(nil), l.lines[n:]...)
		l.base += n
	}
	return len(p), nil
}

func (l *Log) Since(seq int) ([]string, int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	i := max(seq-l.base, 0)
	if i > len(l.lines) {
		i = len(l.lines)
	}
	return append([]string(nil), l.lines[i:]...), l.base + len(l.lines)
}

type Proc struct {
	State   string // loading | ready | stopping | stopped | failed
	Port    int
	Started time.Time
	cmd     *exec.Cmd
	done    chan struct{}
	log     *Log
}

type PplJob struct {
	ID      int                `json:"id"`
	Model   string             `json:"model"`
	Params  map[string]any     `json:"params"`
	State   string             `json:"state"` // running | done | failed
	Started time.Time          `json:"started"`
	Seconds float64            `json:"seconds"`
	PPL     float64            `json:"ppl,omitempty"`
	Domains map[string]float64 `json:"domains,omitempty"`
	Error   string             `json:"error,omitempty"`
	log     *Log
	cmd     *exec.Cmd
}

type Server struct {
	cfgPath string
	mu      sync.Mutex
	cfg     *Config
	procs   map[string]*Proc
	logs    map[string]*Log // per model, survives restarts
	jobs    []*PplJob
	cpuPrev [2]uint64
}

func (s *Server) modelLog(id string) *Log {
	if s.logs[id] == nil {
		s.logs[id] = &Log{}
	}
	return s.logs[id]
}

func freePort(cfg *Config, procs map[string]*Proc) int {
	taken := map[int]bool{}
	for _, m := range cfg.Models {
		taken[m.Port] = true
	}
	for _, p := range procs {
		if p.State != "stopped" && p.State != "failed" {
			taken[p.Port] = true
		}
	}
	for p := cfg.PortStart; ; p++ {
		if !taken[p] {
			return p
		}
	}
}

// load starts a model (no-op when already loading/ready). Caller holds s.mu.
func (s *Server) load(id string) (*Proc, error) {
	m, ok := s.cfg.Models[id]
	if !ok {
		return nil, fmt.Errorf("unknown model %q", id)
	}
	if p := s.procs[id]; p != nil && (p.State == "loading" || p.State == "ready") {
		return p, nil
	}
	if s.cfg.Exclusive {
		for other, p := range s.procs {
			if other != id {
				s.stop(p)
			}
		}
		if j := s.runningJob(); j != nil {
			return nil, fmt.Errorf("perplexity job %d is using the GPU", j.ID)
		}
	}
	port := m.Port
	if port == 0 {
		port = freePort(s.cfg, s.procs)
	}
	lg := s.modelLog(id)
	args := append([]string{m.Path, "--host", "127.0.0.1", "--port", strconv.Itoa(port), "--model-id", id}, m.Args...)
	cmd := exec.Command(s.cfg.ServeBin, args...)
	cmd.Stdout, cmd.Stderr = lg, lg
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	fmt.Fprintf(lg, "\n=== %s  %s %s\n", time.Now().Format(time.TimeOnly), s.cfg.ServeBin, strings.Join(args, " "))
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(lg, "start failed: %v\n", err)
		s.procs[id] = &Proc{State: "failed", Port: port, log: lg}
		return nil, err
	}
	p := &Proc{State: "loading", Port: port, Started: time.Now(), cmd: cmd, done: make(chan struct{}), log: lg}
	s.procs[id] = p
	go func() {
		err := cmd.Wait()
		s.mu.Lock()
		if p.State == "stopping" {
			p.State = "stopped"
		} else {
			p.State = "failed"
		}
		s.mu.Unlock()
		fmt.Fprintf(lg, "=== process exited: %v\n", err)
		close(p.done)
	}()
	go func() {
		health := fmt.Sprintf("http://127.0.0.1:%d/health", port)
		for {
			select {
			case <-p.done:
				return
			case <-time.After(time.Second):
			}
			if r, err := http.Get(health); err == nil {
				r.Body.Close()
				if r.StatusCode == 200 {
					s.mu.Lock()
					if p.State == "loading" {
						p.State = "ready"
						fmt.Fprintf(lg, "=== ready on :%d after %s\n", port, time.Since(p.Started).Round(time.Second))
					}
					s.mu.Unlock()
					return
				}
			}
		}
	}()
	return p, nil
}

// stop signals the process group and waits for exit. Caller holds s.mu; it is released while waiting.
func (s *Server) stop(p *Proc) {
	if p == nil || p.cmd == nil || (p.State != "loading" && p.State != "ready" && p.State != "stopping") {
		return
	}
	p.State = "stopping"
	pgid := -p.cmd.Process.Pid
	syscall.Kill(pgid, syscall.SIGTERM)
	s.mu.Unlock()
	select {
	case <-p.done:
	case <-time.After(30 * time.Second):
		syscall.Kill(pgid, syscall.SIGKILL)
		<-p.done
	}
	s.mu.Lock()
}

func (s *Server) runningJob() *PplJob {
	for _, j := range s.jobs {
		if j.State == "running" {
			return j
		}
	}
	return nil
}

// waitReady blocks until the model serves or fails.
func (s *Server) waitReady(id string) (int, error) {
	s.mu.Lock()
	p, err := s.load(id)
	timeout := time.Duration(s.cfg.LoadTimeout) * time.Second
	s.mu.Unlock()
	if err != nil {
		return 0, err
	}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		st := p.State
		s.mu.Unlock()
		switch st {
		case "ready":
			return p.Port, nil
		case "failed", "stopped":
			return 0, fmt.Errorf("model %q %s while loading (see logs)", id, st)
		}
		time.Sleep(250 * time.Millisecond)
	}
	return 0, fmt.Errorf("model %q not ready after %s", id, timeout)
}

// ---------- HTTP ----------

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func apiErr(w http.ResponseWriter, code int, err error) {
	writeJSON(w, code, map[string]any{"error": map[string]string{"message": err.Error(), "type": "proxy_error"}})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := make([]string, 0, len(s.cfg.Models))
	for id := range s.cfg.Models {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := []map[string]any{}
	for _, id := range ids {
		m := s.cfg.Models[id]
		e := map[string]any{"id": id, "path": m.Path, "args": m.Args, "description": m.Description,
			"state": "stopped", "port": m.Port}
		if p := s.procs[id]; p != nil {
			e["state"], e["port"] = p.State, p.Port
			if p.State == "loading" || p.State == "ready" {
				e["uptime"] = int(time.Since(p.Started).Seconds())
			}
		}
		if st, err := os.Stat(m.Path); err == nil {
			e["size"] = st.Size()
		} else {
			e["missing"] = true
		}
		out = append(out, e)
	}
	writeJSON(w, 200, out)
}

func (s *Server) handleModelAction(w http.ResponseWriter, r *http.Request) {
	id, action := r.PathValue("id"), r.PathValue("action")
	s.mu.Lock()
	defer s.mu.Unlock()
	switch action {
	case "load":
		if _, err := s.load(id); err != nil {
			apiErr(w, 400, err)
			return
		}
	case "unload":
		s.stop(s.procs[id])
	default:
		apiErr(w, 404, errors.New("unknown action"))
		return
	}
	writeJSON(w, 200, map[string]string{"ok": action})
}

func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request) {
	since, _ := strconv.Atoi(r.URL.Query().Get("since"))
	s.mu.Lock()
	var lg *Log
	if id := r.URL.Query().Get("model"); id != "" {
		lg = s.modelLog(id)
	} else if jid, err := strconv.Atoi(r.URL.Query().Get("job")); err == nil {
		for _, j := range s.jobs {
			if j.ID == jid {
				lg = j.log
			}
		}
	}
	s.mu.Unlock()
	if lg == nil {
		apiErr(w, 404, errors.New("no such log"))
		return
	}
	lines, next := lg.Since(since)
	writeJSON(w, 200, map[string]any{"lines": lines, "next": next})
}

// handleModelMetrics returns the ninfer_* gauges of a ready model as a flat map.
func (s *Server) handleModelMetrics(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	p := s.procs[r.PathValue("id")]
	ready := p != nil && p.State == "ready"
	port := 0
	if p != nil {
		port = p.Port
	}
	s.mu.Unlock()
	if !ready {
		writeJSON(w, 200, map[string]float64{})
		return
	}
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/metrics", port))
	if err != nil {
		apiErr(w, 502, err)
		return
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	writeJSON(w, 200, parseProm(string(body)))
}

func parseProm(text string) map[string]float64 {
	out := map[string]float64{}
	for _, line := range strings.Split(text, "\n") {
		f := strings.Fields(line)
		if len(f) < 2 || strings.HasPrefix(line, "#") {
			continue
		}
		if v, err := strconv.ParseFloat(f[1], 64); err == nil {
			out[f[0]] = v
		}
	}
	return out
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	st := map[string]any{}
	if b, err := os.ReadFile("/proc/stat"); err == nil {
		f := strings.Fields(strings.SplitN(string(b), "\n", 2)[0])[1:]
		var total, idle uint64
		for i, v := range f {
			n, _ := strconv.ParseUint(v, 10, 64)
			total += n
			if i == 3 || i == 4 { // idle + iowait
				idle += n
			}
		}
		s.mu.Lock()
		dt, di := total-s.cpuPrev[0], idle-s.cpuPrev[1]
		s.cpuPrev = [2]uint64{total, idle}
		s.mu.Unlock()
		if dt > 0 && dt != total {
			st["cpu"] = 100 * float64(dt-di) / float64(dt)
		}
	}
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		mem := map[string]float64{}
		for _, line := range strings.Split(string(b), "\n") {
			if f := strings.Fields(line); len(f) >= 2 {
				v, _ := strconv.ParseFloat(f[1], 64)
				mem[strings.TrimSuffix(f[0], ":")] = v * 1024
			}
		}
		st["mem_total"], st["mem_used"] = mem["MemTotal"], mem["MemTotal"]-mem["MemAvailable"]
	}
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		st["load"] = strings.Join(strings.Fields(string(b))[:3], " ")
	}
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
		"--format=csv,noheader,nounits").Output()
	if err == nil {
		gpus := []map[string]any{}
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			f := strings.Split(line, ", ")
			if len(f) < 7 {
				continue
			}
			num := func(i int) float64 { v, _ := strconv.ParseFloat(strings.TrimSpace(f[i]), 64); return v }
			gpus = append(gpus, map[string]any{"name": f[0], "util": num(1), "mem_used": num(2) * (1 << 20),
				"mem_total": num(3) * (1 << 20), "temp": num(4), "power": num(5), "power_limit": num(6)})
		}
		st["gpus"] = gpus
	}
	writeJSON(w, 200, st)
}

func (s *Server) handleConfig(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		b, err := os.ReadFile(s.cfgPath)
		if err != nil {
			apiErr(w, 500, err)
			return
		}
		w.Header().Set("Content-Type", "text/yaml; charset=utf-8")
		w.Write(b)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		apiErr(w, 400, err)
		return
	}
	cfg, err := parseConfig(body)
	if err != nil {
		apiErr(w, 400, err)
		return
	}
	// WriteFile truncates in place, so a single-file docker bind mount keeps working.
	if err := os.WriteFile(s.cfgPath, body, 0o644); err != nil {
		apiErr(w, 500, err)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, p := range s.procs {
		if _, ok := cfg.Models[id]; !ok {
			s.stop(p) // removed from config
			delete(s.procs, id)
		}
	}
	if cfg.Listen != s.cfg.Listen {
		log.Printf("listen change to %s needs a restart", cfg.Listen)
	}
	s.cfg = cfg
	writeJSON(w, 200, map[string]string{"ok": "saved"})
}

func (s *Server) handlePpl(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		s.mu.Lock()
		defer s.mu.Unlock()
		writeJSON(w, 200, s.jobs)
		return
	}
	var req struct {
		Model   string `json:"model"`
		Mode    string `json:"mode"` // quick | full | text
		Text    string `json:"text"`
		Context int    `json:"context"`
		Stride  int    `json:"stride"`
		KV      string `json:"kv"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<20)).Decode(&req); err != nil {
		apiErr(w, 400, err)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if r.URL.Query().Get("cancel") != "" {
		if j := s.runningJob(); j != nil && j.cmd != nil {
			syscall.Kill(-j.cmd.Process.Pid, syscall.SIGTERM)
		}
		writeJSON(w, 200, map[string]string{"ok": "cancel"})
		return
	}
	m, ok := s.cfg.Models[req.Model]
	if !ok {
		apiErr(w, 400, fmt.Errorf("unknown model %q", req.Model))
		return
	}
	if j := s.runningJob(); j != nil {
		apiErr(w, 409, fmt.Errorf("job %d still running", j.ID))
		return
	}
	dir, err := os.MkdirTemp("", "ninfer-ppl-")
	if err != nil {
		apiErr(w, 500, err)
		return
	}
	out := filepath.Join(dir, "out")
	args := []string{m.Path, "--output", out}
	switch req.Mode {
	case "text":
		if strings.TrimSpace(req.Text) == "" {
			apiErr(w, 400, errors.New("text is empty"))
			return
		}
		tf := filepath.Join(dir, "input.txt")
		os.WriteFile(tf, []byte(req.Text), 0o644)
		args = append(args, "--text", tf)
	case "full":
		args = append(args, "--corpus", s.cfg.Corpus)
	default:
		args = append(args, "--corpus", s.cfg.Corpus, "--quick")
	}
	if req.Context > 0 {
		args = append(args, "--context", strconv.Itoa(req.Context))
	}
	if req.Stride > 0 {
		args = append(args, "--stride", strconv.Itoa(req.Stride))
	}
	if req.KV != "" {
		args = append(args, "--kv-dtype", req.KV)
	}
	if s.cfg.Exclusive { // one GPU: perplexity needs the memory the servers hold
		for _, p := range s.procs {
			s.stop(p)
		}
	}
	j := &PplJob{ID: len(s.jobs) + 1, Model: req.Model, State: "running", Started: time.Now(), log: &Log{},
		Params: map[string]any{"mode": req.Mode, "context": req.Context, "stride": req.Stride, "kv": req.KV}}
	fmt.Fprintf(j.log, "=== %s %s\n", s.cfg.PplBin, strings.Join(args, " "))
	cmd := exec.Command(s.cfg.PplBin, args...)
	cmd.Stdout, cmd.Stderr = j.log, j.log
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		apiErr(w, 500, err)
		return
	}
	j.cmd = cmd
	s.jobs = append(s.jobs, j)
	go func() {
		err := cmd.Wait()
		rep, rerr := readReport(filepath.Join(out, "report.json"))
		s.mu.Lock()
		defer s.mu.Unlock()
		j.Seconds = time.Since(j.Started).Seconds()
		switch {
		case err != nil:
			j.State, j.Error = "failed", err.Error()
		case rerr != nil:
			j.State, j.Error = "failed", rerr.Error()
		default:
			j.State, j.PPL, j.Domains = "done", rep.Overall.Perplexity, map[string]float64{}
			for _, d := range rep.Domains {
				j.Domains[d.Domain] = d.Perplexity
			}
		}
		fmt.Fprintf(j.log, "=== %s %s\n", j.State, j.Error)
		os.RemoveAll(dir) // ponytail: report.json is dropped after parsing; persist it if runs need archiving
	}()
	writeJSON(w, 200, j)
}

type pplReport struct {
	Overall struct {
		Perplexity float64 `json:"perplexity"`
	} `json:"overall"`
	Domains []struct {
		Domain     string  `json:"domain"`
		Perplexity float64 `json:"perplexity"`
	} `json:"domains"`
}

func readReport(path string) (*pplReport, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rep pplReport
	return &rep, json.Unmarshal(b, &rep)
}

// handleOpenAI routes /v1/* by the "model" field of the request body, loading the model on demand.
func (s *Server) handleOpenAI(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
		s.mu.Lock()
		data := []map[string]any{}
		for id := range s.cfg.Models {
			state := "stopped"
			if p := s.procs[id]; p != nil {
				state = p.State
			}
			data = append(data, map[string]any{"id": id, "object": "model", "owned_by": "ninfer", "meta": map[string]any{"state": state}})
		}
		s.mu.Unlock()
		sort.Slice(data, func(a, b int) bool { return data[a]["id"].(string) < data[b]["id"].(string) })
		writeJSON(w, 200, map[string]any{"object": "list", "data": data})
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		apiErr(w, 400, err)
		return
	}
	var probe struct {
		Model string `json:"model"`
	}
	json.Unmarshal(body, &probe)
	id := probe.Model
	if id == "" {
		id = strings.TrimPrefix(r.URL.Path, "/v1/models/") // GET /v1/models/{id}
	}
	s.mu.Lock()
	_, known := s.cfg.Models[id]
	s.mu.Unlock()
	if !known {
		apiErr(w, 404, fmt.Errorf("model %q is not in config.yaml", id))
		return
	}
	port, err := s.waitReady(id)
	if err != nil {
		apiErr(w, 503, err)
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	target, _ := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", port))
	rp := httputil.NewSingleHostReverseProxy(target)
	rp.FlushInterval = -1 // stream SSE immediately
	rp.ServeHTTP(w, r)
}

func main() {
	cfgPath := flag.String("config", "config.yaml", "path to config.yaml")
	flag.Parse()
	b, err := os.ReadFile(*cfgPath)
	if err != nil {
		log.Fatal(err)
	}
	cfg, err := parseConfig(b)
	if err != nil {
		log.Fatalf("%s: %v", *cfgPath, err)
	}
	s := &Server{cfgPath: *cfgPath, cfg: cfg, procs: map[string]*Proc{}, logs: map[string]*Log{}}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	mux.HandleFunc("GET /api/models", s.handleModels)
	mux.HandleFunc("POST /api/models/{id}/{action}", s.handleModelAction)
	mux.HandleFunc("GET /api/models/{id}/metrics", s.handleModelMetrics)
	mux.HandleFunc("GET /api/logs", s.handleLogs)
	mux.HandleFunc("GET /api/stats", s.handleStats)
	mux.HandleFunc("/api/config", s.handleConfig)
	mux.HandleFunc("/api/ppl", s.handlePpl)
	mux.HandleFunc("/v1/", s.handleOpenAI)

	// Stop children on SIGTERM/SIGINT so GPU memory is released with the container.
	go func() {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch, syscall.SIGTERM, syscall.SIGINT)
		<-ch
		s.mu.Lock()
		for _, p := range s.procs {
			s.stop(p)
		}
		if j := s.runningJob(); j != nil && j.cmd != nil {
			syscall.Kill(-j.cmd.Process.Pid, syscall.SIGTERM)
		}
		os.Exit(0)
	}()

	log.Printf("ninfer-proxy on %s (%d models, config %s)", cfg.Listen, len(cfg.Models), *cfgPath)
	log.Fatal(http.ListenAndServe(cfg.Listen, mux))
}
