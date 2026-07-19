package cli

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"
)

// captureStdout runs fn with os.Stdout redirected to a pipe and returns what
// was written (doctor prints via fmt.Println straight to os.Stdout).
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	orig := os.Stdout
	os.Stdout = w
	fn()
	os.Stdout = orig
	_ = w.Close()
	out, _ := io.ReadAll(r)
	return string(out)
}

func runDoctor(t *testing.T, args ...string) (string, error) {
	t.Helper()
	store := newCmdTestStore(t)
	cmd := newDoctorCmd()
	// The test store's stub embedder is 16-dim; align the config so the
	// embed-dim guardrail sees a consistent setup (the default is 384).
	cfg := defaultConfig()
	cfg.EmbedDim = 16
	ctx := context.WithValue(context.Background(), storeKey, store)
	ctx = context.WithValue(ctx, cfgKey, cfg)
	cmd.SetArgs(args)
	cmd.SilenceUsage = true
	var err error
	out := captureStdout(t, func() { err = cmd.ExecuteContext(ctx) })
	return out, err
}

func TestDoctorReportsReady(t *testing.T) {
	out, err := runDoctor(t)
	if err != nil {
		t.Fatalf("doctor should succeed: %v\n%s", err, out)
	}
	if !strings.Contains(out, "OK — ready") {
		t.Errorf("expected ready banner, got:\n%s", out)
	}
}

func TestDoctorJSON(t *testing.T) {
	out, err := runDoctor(t, "--json")
	if err != nil {
		t.Fatalf("doctor --json should succeed: %v\n%s", err, out)
	}
	start := strings.Index(out, "{")
	if start < 0 {
		t.Fatalf("no JSON object in output:\n%s", out)
	}
	var report struct {
		OK     bool `json:"ok"`
		Checks []struct {
			Name string `json:"name"`
			OK   bool   `json:"ok"`
		} `json:"checks"`
	}
	if err := json.Unmarshal([]byte(out[start:]), &report); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out)
	}
	if !report.OK {
		t.Errorf("report should be ok: %+v", report)
	}
	names := map[string]bool{}
	for _, c := range report.Checks {
		names[c.Name] = true
	}
	for _, want := range []string{"config", "store", "embedder"} {
		if !names[want] {
			t.Errorf("missing check %q in %v", want, names)
		}
	}
}
