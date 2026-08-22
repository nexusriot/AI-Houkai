package memory

import (
	"compress/gzip"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestJournalRotateRenamesBeforeCompressing(t *testing.T) {
	dir := t.TempDir()
	j := NewJournal(filepath.Join(dir, "j.log"), 64, 90)
	for i := 0; i < 5; i++ {
		j.Append(JournalEntry{TS: 1000 + float64(i), Op: "remember", ID: "id", Meta: map[string]any{}})
	}
	if err := j.rotate(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(j.Path); !os.IsNotExist(err) {
		t.Fatal("active file must be renamed away, not truncated in place")
	}
	archives, _ := filepath.Glob(filepath.Join(dir, "j-*.log.gz"))
	if len(archives) != 1 {
		t.Fatalf("archives = %v, want exactly one", archives)
	}
	if leftovers, _ := filepath.Glob(filepath.Join(dir, "j-*.log")); len(leftovers) != 0 {
		t.Fatalf("rotated plain file not cleaned up: %v", leftovers)
	}
	entries, err := j.Read(ReadOpts{IncludeArchives: true})
	if err != nil || len(entries) != 5 {
		t.Fatalf("Read after rotate: %d entries err=%v, want 5", len(entries), err)
	}
}

func TestJournalTruncatedArchiveTolerated(t *testing.T) {
	dir := t.TempDir()
	j := NewJournal(filepath.Join(dir, "j.log"), 64, 90)
	for i := 0; i < 5; i++ {
		j.Append(JournalEntry{TS: 1000 + float64(i), Op: "remember", ID: "old", Meta: map[string]any{}})
	}
	if err := j.rotate(); err != nil {
		t.Fatal(err)
	}
	archive, _ := filepath.Glob(filepath.Join(dir, "j-*.log.gz"))
	raw, _ := os.ReadFile(archive[0])
	os.WriteFile(archive[0], raw[:len(raw)/2], 0o644)

	for i := 0; i < 3; i++ {
		j.Append(JournalEntry{TS: 2000 + float64(i), Op: "remember", ID: "new", Meta: map[string]any{}})
	}
	entries, err := j.Read(ReadOpts{IncludeArchives: true})
	if err != nil {
		t.Fatalf("Read must tolerate a truncated archive: %v", err)
	}
	live := 0
	for _, e := range entries {
		if e.ID == "new" {
			live++
		}
	}
	if live != 3 {
		t.Fatalf("active entries lost behind a truncated archive: %d/3", live)
	}
}

func TestJournalPlainRotationWinsOverSameStemArchive(t *testing.T) {
	dir := t.TempDir()
	j := NewJournal(filepath.Join(dir, "j.log"), 64, 90)
	for i := 0; i < 4; i++ {
		j.Append(JournalEntry{TS: 1000 + float64(i), Op: "remember", ID: "id", Meta: map[string]any{}})
	}
	if err := j.rotate(); err != nil {
		t.Fatal(err)
	}
	// Recreate the crash-between-compress-and-unlink shape: both files exist.
	archives, _ := filepath.Glob(filepath.Join(dir, "j-*.log.gz"))
	f, _ := os.Open(archives[0])
	gr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	buf := make([]byte, 4096)
	for {
		n, rerr := gr.Read(buf)
		sb.Write(buf[:n])
		if rerr != nil {
			break
		}
	}
	f.Close()
	plain := strings.TrimSuffix(archives[0], ".gz")
	os.WriteFile(plain, []byte(sb.String()), 0o644)

	entries, err := j.Read(ReadOpts{IncludeArchives: true})
	if err != nil || len(entries) != 4 {
		t.Fatalf("entries = %d err=%v, want 4 (no double-yield)", len(entries), err)
	}
}
