package cli

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// A nil slice marshals to `null`, which is valid JSON but not iterable: on a
// fresh store `houkai list -f json | jq '.[]'` failed with "Cannot iterate over
// null". Machine formats must always emit a well-formed document.
func TestPrintRowsEmptyJSONIsAnArray(t *testing.T) {
	var buf bytes.Buffer
	PrintRows(&buf, nil, FormatJSON)

	var rows []MemRow
	if err := json.Unmarshal(buf.Bytes(), &rows); err != nil {
		t.Fatalf("unmarshal %q: %v", buf.String(), err)
	}
	if len(rows) != 0 {
		t.Fatalf("rows = %+v, want empty", rows)
	}
	if got := strings.TrimSpace(buf.String()); got != "[]" {
		t.Fatalf("output = %q, want []", got)
	}
}

func TestPrintRowsEmptyTSVKeepsTheHeader(t *testing.T) {
	var buf bytes.Buffer
	PrintRows(&buf, nil, FormatTSV)

	line := strings.TrimSpace(buf.String())
	if !strings.HasPrefix(line, "id\ttype\t") {
		t.Fatalf("output = %q, want the TSV header", line)
	}
}

func TestPrintRowsJSONShapeIsUnchangedForRealRows(t *testing.T) {
	var buf bytes.Buffer
	PrintRows(&buf, []MemRow{{ID: "abc", Text: "hi"}}, FormatJSON)

	var rows []MemRow
	if err := json.Unmarshal(buf.Bytes(), &rows); err != nil {
		t.Fatalf("unmarshal %q: %v", buf.String(), err)
	}
	if len(rows) != 1 || rows[0].ID != "abc" || rows[0].Text != "hi" {
		t.Fatalf("rows = %+v", rows)
	}
}
