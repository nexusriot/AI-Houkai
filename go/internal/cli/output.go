package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

var (
	headerStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("12"))
	idStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	tagStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("10"))
	impStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))
	dimStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
)

// OutputFormat controls how results are rendered.
type OutputFormat string

const (
	FormatAuto OutputFormat = "auto"
	FormatJSON OutputFormat = "json"
	FormatTSV  OutputFormat = "tsv"
)

func isTTY() bool {
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

func fmtAge(ts float64) string {
	t := time.Unix(int64(ts), 0)
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	case d < 7*24*time.Hour:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	default:
		return t.Format("2006-01-02")
	}
}

func fmtID(id string) string {
	if len(id) >= 8 {
		return id[:8]
	}
	return id
}

// MemRow is a generic renderable row.
type MemRow struct {
	ID           string
	Text         string
	Type         string
	Tags         []string
	Importance   float32
	Score        float32
	CreatedAt    float64
	SupersededBy string
}

func PrintRows(w io.Writer, rows []MemRow, format OutputFormat) {
	switch format {
	case FormatJSON:
		b, _ := json.MarshalIndent(rows, "", "  ")
		fmt.Fprintln(w, string(b))
	case FormatTSV:
		fmt.Fprintln(w, "id\ttype\timportance\tcreated_at\ttags\ttext")
		for _, r := range rows {
			fmt.Fprintf(w, "%s\t%s\t%.2f\t%s\t%s\t%s\n",
				r.ID, r.Type, r.Importance,
				time.Unix(int64(r.CreatedAt), 0).Format(time.RFC3339),
				strings.Join(r.Tags, ","),
				strings.ReplaceAll(r.Text, "\t", " "))
		}
	default: // auto
		if isTTY() {
			printRich(w, rows)
		} else {
			PrintRows(w, rows, FormatTSV)
		}
	}
}

func printRich(w io.Writer, rows []MemRow) {
	colW := []int{8, 11, 14, 10, 16, 0} // id, type, importance, age, tags, text
	hdr := []string{"ID", "TYPE", "IMPORTANCE", "AGE", "TAGS", "TEXT"}
	printRow(w, hdr, colW, headerStyle)
	fmt.Fprintln(w, strings.Repeat("─", 80))
	for _, r := range rows {
		sup := ""
		if r.SupersededBy != "" {
			sup = " [superseded]"
		}
		cells := []string{
			idStyle.Render(fmtID(r.ID)),
			r.Type,
			impStyle.Render(fmt.Sprintf("%.2f", r.Importance)),
			dimStyle.Render(fmtAge(r.CreatedAt)),
			tagStyle.Render(strings.Join(r.Tags, ",")),
			r.Text + dimStyle.Render(sup),
		}
		printRow(w, cells, colW, lipgloss.NewStyle())
	}
}

func printRow(w io.Writer, cells []string, widths []int, _ lipgloss.Style) {
	for i, c := range cells {
		if i < len(widths)-1 && widths[i] > 0 {
			fmt.Fprintf(w, "%-*s  ", widths[i], truncate(stripAnsi(c), widths[i]))
		} else {
			fmt.Fprintf(w, "%s", c)
		}
	}
	fmt.Fprintln(w)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

func stripAnsi(s string) string {
	// Very basic ANSI escape stripper for width computation.
	out := strings.Builder{}
	inEsc := false
	for _, r := range s {
		if inEsc {
			if r == 'm' {
				inEsc = false
			}
			continue
		}
		if r == '\x1b' {
			inEsc = true
			continue
		}
		out.WriteRune(r)
	}
	return out.String()
}

func Confirm(prompt string) bool {
	fmt.Printf("%s [y/N] ", prompt)
	var resp string
	_, _ = fmt.Scanln(&resp)
	return strings.ToLower(strings.TrimSpace(resp)) == "y"
}
