// Append-only audit journal for the memory store.
//
// One JSON object per line, gzipped on rotation. Writes are best-effort:
// a failure to journal never breaks the underlying memory operation. Use
// Journal.Read(...) to stream historical entries with filtering and
// MemoryStore.Undo(entry) to reverse a single operation.
package memory

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// JournalOp tags the kind of mutation that produced an entry. Free-form
// string so callers (reflect/decay/import) can introduce new ops.
type JournalOp = string

// JournalEntry is one line in the journal.
type JournalEntry struct {
	TS     float64                `json:"ts"`
	Op     JournalOp              `json:"op"`
	Actor  string                 `json:"actor"`
	ID     string                 `json:"id"`
	Before map[string]any         `json:"before,omitempty"`
	After  map[string]any         `json:"after,omitempty"`
	Meta   map[string]any         `json:"meta"`
}

// Summary returns a short human-readable description of the entry.
func (e JournalEntry) Summary() string {
	short := func(s string) string {
		if len(s) > 8 {
			return s[:8]
		}
		return s
	}
	clip := func(s string) string {
		if len(s) > 60 {
			return s[:60]
		}
		return s
	}
	switch e.Op {
	case "remember":
		if e.After != nil {
			if t, _ := e.After["text"].(string); t != "" {
				return fmt.Sprintf("remember %s «%s»", short(e.ID), clip(t))
			}
		}
	case "forget":
		if e.Before != nil {
			if t, _ := e.Before["text"].(string); t != "" {
				return fmt.Sprintf("forget %s «%s»", short(e.ID), clip(t))
			}
		}
	case "supersede":
		if e.Meta != nil {
			if nid, _ := e.Meta["new_id"].(string); nid != "" {
				return fmt.Sprintf("supersede %s → %s", short(e.ID), short(nid))
			}
		}
	case "link", "unlink":
		dst, _ := e.Meta["dst_id"].(string)
		rel, _ := e.Meta["rel"].(string)
		return fmt.Sprintf("%s %s → %s (%s)", e.Op, short(e.ID), short(dst), rel)
	}
	return fmt.Sprintf("%s %s", e.Op, short(e.ID))
}

func (e JournalEntry) toLine() ([]byte, error) {
	b, err := json.Marshal(e)
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// Journal is an append-only JSONL log with size-based gzip rotation.
//
// Writes are atomic at the line level on POSIX (O_APPEND linearises across
// processes for writes ≤ PIPE_BUF). Reads tolerate truncated last lines (a
// crash mid-write) by skipping un-parseable JSON.
type Journal struct {
	Path             string
	RotateMB         int
	KeepDays         int
	writesSinceCheck int
}

const rotateCheckEvery = 256

// NewJournal constructs a Journal. The on-disk file is created lazily.
func NewJournal(path string, rotateMB, keepDays int) *Journal {
	if rotateMB <= 0 {
		rotateMB = 64
	}
	if keepDays <= 0 {
		keepDays = 90
	}
	return &Journal{Path: path, RotateMB: rotateMB, KeepDays: keepDays}
}

// Append writes one entry. All errors are logged, never returned, so a
// journal failure cannot break the underlying memory op.
func (j *Journal) Append(e JournalEntry) {
	if j == nil || j.Path == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(j.Path), 0o755); err != nil {
		log.Printf("ai-houkai journal mkdir: %v", err)
		return
	}
	line, err := e.toLine()
	if err != nil {
		log.Printf("ai-houkai journal marshal: %v", err)
		return
	}
	f, err := os.OpenFile(j.Path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Printf("ai-houkai journal open: %v", err)
		return
	}
	if _, err := f.Write(line); err != nil {
		log.Printf("ai-houkai journal write: %v", err)
	}
	_ = f.Close()

	j.writesSinceCheck++
	if j.writesSinceCheck >= rotateCheckEvery {
		j.writesSinceCheck = 0
		if err := j.maybeRotate(); err != nil {
			log.Printf("ai-houkai journal rotate: %v", err)
		}
	}
}

func (j *Journal) maybeRotate() error {
	if j.RotateMB > 0 {
		st, err := os.Stat(j.Path)
		if err == nil && st.Size() >= int64(j.RotateMB)*1024*1024 {
			if err := j.rotate(); err != nil {
				return err
			}
		}
	}
	if j.KeepDays > 0 {
		j.pruneArchives()
	}
	return nil
}

func (j *Journal) rotate() error {
	stamp := time.Now().UTC().Format("20060102T150405")
	stem := strings.TrimSuffix(filepath.Base(j.Path), filepath.Ext(j.Path))
	dir := filepath.Dir(j.Path)
	archive := filepath.Join(dir, fmt.Sprintf("%s-%s.log.gz", stem, stamp))
	if _, err := os.Stat(archive); err == nil {
		archive = filepath.Join(dir, fmt.Sprintf("%s-%s-%d.log.gz", stem, stamp, os.Getpid()))
	}
	src, err := os.Open(j.Path)
	if err != nil {
		return err
	}
	defer src.Close()
	dst, err := os.Create(archive)
	if err != nil {
		return err
	}
	gw := gzip.NewWriter(dst)
	if _, err := io.Copy(gw, src); err != nil {
		_ = gw.Close()
		_ = dst.Close()
		return err
	}
	if err := gw.Close(); err != nil {
		_ = dst.Close()
		return err
	}
	if err := dst.Close(); err != nil {
		return err
	}
	// Truncate in place so any concurrent writer's fd remains valid.
	return os.Truncate(j.Path, 0)
}

func (j *Journal) pruneArchives() {
	stem := strings.TrimSuffix(filepath.Base(j.Path), filepath.Ext(j.Path))
	pattern := filepath.Join(filepath.Dir(j.Path), stem+"-*.log.gz")
	matches, _ := filepath.Glob(pattern)
	cutoff := time.Now().Add(-time.Duration(j.KeepDays) * 24 * time.Hour)
	for _, p := range matches {
		st, err := os.Stat(p)
		if err == nil && st.ModTime().Before(cutoff) {
			_ = os.Remove(p)
		}
	}
}

// ReadOpts filters Journal.Read output.
type ReadOpts struct {
	Since           float64
	Until           float64
	Op              string
	Actor           string
	MemoryID        string
	Limit           int
	IncludeArchives bool
}

// Read streams matching entries oldest first.
func (j *Journal) Read(opts ReadOpts) ([]JournalEntry, error) {
	var files []string
	if opts.IncludeArchives {
		stem := strings.TrimSuffix(filepath.Base(j.Path), filepath.Ext(j.Path))
		pattern := filepath.Join(filepath.Dir(j.Path), stem+"-*.log.gz")
		archives, _ := filepath.Glob(pattern)
		sort.Strings(archives)
		files = append(files, archives...)
	}
	if _, err := os.Stat(j.Path); err == nil {
		files = append(files, j.Path)
	}

	var out []JournalEntry
	for _, fp := range files {
		f, err := os.Open(fp)
		if err != nil {
			continue
		}
		var r io.Reader = f
		if strings.HasSuffix(fp, ".gz") {
			gr, err := gzip.NewReader(f)
			if err != nil {
				_ = f.Close()
				continue
			}
			defer gr.Close()
			r = gr
		}
		sc := bufio.NewScanner(r)
		sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			var e JournalEntry
			if err := json.Unmarshal([]byte(line), &e); err != nil {
				continue
			}
			if opts.Since != 0 && e.TS < opts.Since {
				continue
			}
			if opts.Until != 0 && e.TS > opts.Until {
				continue
			}
			if opts.Op != "" && e.Op != opts.Op {
				continue
			}
			if opts.Actor != "" && e.Actor != opts.Actor {
				continue
			}
			if opts.MemoryID != "" && e.ID != opts.MemoryID {
				continue
			}
			out = append(out, e)
			if opts.Limit > 0 && len(out) >= opts.Limit {
				_ = f.Close()
				return out, nil
			}
		}
		_ = f.Close()
	}
	return out, nil
}

// FindByTS returns the entry whose timestamp matches ts within tol (seconds).
func (j *Journal) FindByTS(ts, tol float64) (*JournalEntry, error) {
	if tol <= 0 {
		tol = 1e-3
	}
	entries, err := j.Read(ReadOpts{
		Since: ts - tol, Until: ts + tol, IncludeArchives: true,
	})
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if math.Abs(e.TS-ts) <= tol {
			return &e, nil
		}
	}
	return nil, nil
}
