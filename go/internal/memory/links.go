package memory

// Standard relation vocabulary.
const (
	RelSupersedes   = "supersedes"
	RelRefines      = "refines"
	RelDerivedFrom  = "derived_from"
	RelExampleOf    = "example_of"
	RelContradicts  = "contradicts"
	RelRelated      = "related"
)

// addLink appends a link to m if not already present (idempotent by to+rel).
func addLink(m *Memory, to, rel string) {
	for _, l := range m.Links {
		if l.To == to && l.Rel == rel {
			return
		}
	}
	m.Links = append(m.Links, Link{To: to, Rel: rel})
}

// removeLinks removes links matching rel (empty rel = remove all to dst).
func removeLinks(m *Memory, to, rel string) int {
	keep := m.Links[:0]
	removed := 0
	for _, l := range m.Links {
		if l.To == to && (rel == "" || l.Rel == rel) {
			removed++
		} else {
			keep = append(keep, l)
		}
	}
	m.Links = keep
	return removed
}
