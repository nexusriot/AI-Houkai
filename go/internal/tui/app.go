package tui

// houkai tui — Bubble Tea memory browser with link-graph navigation.
//
// Keys
//	/          focus the search box (semantic recall)
//	escape     clear search / unfocus
//	enter      (in search box) run the search
//	n          open the selected memory's neighbors (walk the graph)
//	b          back one view (breadcrumb stack)
//	r          reload the recent view
//	X          nuke the whole collection (press twice to confirm)
//	q          quit

import (
	"context"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/table"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/nexusriot/ai-houkai/internal/memory"
)

var (
	crumbStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Padding(0, 1)
	titleStyle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("12")).Padding(0, 1)
	detailStyle = lipgloss.NewStyle().BorderStyle(lipgloss.NormalBorder()).
			BorderLeft(true).BorderForeground(lipgloss.Color("12")).PaddingLeft(1)
	statusStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("11")).Padding(0, 1)
	helpStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("8")).Padding(0, 1)
)

// Model is the Bubble Tea model for the memory browser.
type Model struct {
	store      *memory.MemoryStore
	nav        *Navigator
	collection string

	tbl         table.Model
	detail      viewport.Model
	search      textinput.Model
	searching   bool
	nukePending bool
	status      string
	width       int
	height      int
	ready       bool
}

// New builds the TUI model around an open MemoryStore.
func New(store *memory.MemoryStore, collection string) *Model {
	cols := []table.Column{
		{Title: "ID", Width: 8},
		{Title: "TYPE", Width: 10},
		{Title: "IMP", Width: 5},
		{Title: "AGE", Width: 10},
		{Title: "REL/SCORE", Width: 10},
		{Title: "TEXT", Width: 60},
	}
	tbl := table.New(table.WithColumns(cols), table.WithFocused(true))
	st := table.DefaultStyles()
	st.Header = st.Header.Bold(true).Foreground(lipgloss.Color("14"))
	st.Selected = st.Selected.Foreground(lipgloss.Color("0")).Background(lipgloss.Color("14"))
	tbl.SetStyles(st)

	search := textinput.New()
	search.Placeholder = "semantic search… (enter to run, esc to close)"

	return &Model{
		store:      store,
		nav:        NewNavigator(store),
		collection: collection,
		tbl:        tbl,
		detail:     viewport.New(40, 20),
		search:     search,
	}
}

// Run starts the program (full-screen).
func (m *Model) Run() error {
	_, err := tea.NewProgram(m, tea.WithAltScreen()).Run()
	return err
}

func (m *Model) Init() tea.Cmd {
	if _, err := m.nav.OpenRecent(context.Background(), 200); err != nil {
		m.status = err.Error()
	}
	m.showView(m.nav.Current())
	return nil
}

func (m *Model) showView(v View) {
	rows := make([]table.Row, len(v.Rows))
	for i, r := range v.Rows {
		rows[i] = table.Row{r.ID8, r.Type, r.Imp, r.Age, r.Extra, r.Snippet}
	}
	m.tbl.SetRows(rows)
	m.tbl.SetCursor(0)
	m.updateDetail()
}

func (m *Model) selectedMemory() (memory.Memory, bool) {
	row := m.tbl.SelectedRow()
	if row == nil {
		return memory.Memory{}, false
	}
	mem, ok := m.nav.Current().Memories[row[0]]
	return mem, ok
}

func (m *Model) updateDetail() {
	if mem, ok := m.selectedMemory(); ok {
		m.detail.SetContent(DetailText(mem))
		m.detail.GotoTop()
	} else {
		m.detail.SetContent("nothing here")
	}
}

func (m *Model) layout() {
	listW := m.width * 2 / 3
	detailW := m.width - listW - 3
	if detailW < 10 {
		detailW = 10
	}
	bodyH := m.height - 4 // crumb + title + status + help
	if bodyH < 3 {
		bodyH = 3
	}
	m.tbl.SetWidth(listW)
	m.tbl.SetHeight(bodyH)
	textW := listW - (8 + 10 + 5 + 10 + 10) - 12
	if textW < 10 {
		textW = 10
	}
	cols := m.tbl.Columns()
	cols[len(cols)-1].Width = textW
	m.tbl.SetColumns(cols)
	m.detail.Width = detailW
	m.detail.Height = bodyH
	m.search.Width = m.width - 4
	m.ready = true
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.layout()
		return m, nil

	case tea.KeyMsg:
		if m.searching {
			switch msg.String() {
			case "esc":
				m.searching = false
				m.search.SetValue("")
				m.search.Blur()
				return m, nil
			case "enter":
				query := strings.TrimSpace(m.search.Value())
				m.searching = false
				m.search.SetValue("")
				m.search.Blur()
				if query != "" {
					if _, err := m.nav.OpenSearch(context.Background(), query); err != nil {
						m.status = err.Error()
					} else {
						m.status = ""
					}
					m.showView(m.nav.Current())
				}
				return m, nil
			}
			var cmd tea.Cmd
			m.search, cmd = m.search.Update(msg)
			return m, cmd
		}

		// Any key other than a second "X" cancels a pending nuke.
		if msg.String() != "X" {
			m.nukePending = false
		}

		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		case "/":
			m.searching = true
			m.search.Focus()
			return m, textinput.Blink
		case "n":
			if mem, ok := m.selectedMemory(); ok {
				v, err := m.nav.OpenNeighbors(context.Background(), mem)
				if err != nil {
					m.status = err.Error()
					return m, nil
				}
				if len(v.Rows) == 0 {
					m.nav.Pop()
					m.status = "No links on this memory."
					return m, nil
				}
				m.status = ""
				m.showView(v)
			}
			return m, nil
		case "b":
			m.status = ""
			m.showView(m.nav.Back())
			return m, nil
		case "r":
			if _, err := m.nav.OpenRecent(context.Background(), 200); err != nil {
				m.status = err.Error()
			} else {
				m.status = ""
			}
			m.showView(m.nav.Current())
			return m, nil
		case "X":
			return m.handleNuke()
		}
	}

	var cmd tea.Cmd
	m.tbl, cmd = m.tbl.Update(msg)
	m.updateDetail()
	return m, cmd
}

// handleNuke implements the two-press collection wipe: the first X arms it,
// the second confirms and deletes everything, then reloads the recent view.
func (m *Model) handleNuke() (tea.Model, tea.Cmd) {
	ctx := context.Background()
	if m.nukePending {
		m.nukePending = false
		deleted, err := m.store.Nuke(ctx)
		if err != nil {
			m.status = err.Error()
			return m, nil
		}
		if _, err := m.nav.OpenRecent(ctx, 200); err != nil {
			m.status = err.Error()
		} else {
			m.status = fmt.Sprintf("Nuked %d memories.", deleted)
		}
		m.showView(m.nav.Current())
		return m, nil
	}
	count, err := m.store.Count(ctx)
	if err != nil {
		m.status = err.Error()
		return m, nil
	}
	if count == 0 {
		m.status = "Collection is already empty."
		return m, nil
	}
	m.nukePending = true
	m.status = fmt.Sprintf("About to nuke all %d memories. Press X again to confirm.", count)
	return m, nil
}

func (m *Model) View() string {
	if !m.ready {
		return "loading…"
	}
	title := titleStyle.Render(fmt.Sprintf("AI-Houkai — %s", m.collection))
	crumb := crumbStyle.Render(m.nav.Breadcrumb())
	body := lipgloss.JoinHorizontal(lipgloss.Top,
		m.tbl.View(),
		detailStyle.Render(m.detail.View()),
	)
	var bottom string
	if m.searching {
		bottom = m.search.View()
	} else {
		bottom = helpStyle.Render("/ search · n neighbors · b back · r recent · X nuke · q quit")
	}
	status := statusStyle.Render(m.status)
	return lipgloss.JoinVertical(lipgloss.Left, title, crumb, body, status, bottom)
}
