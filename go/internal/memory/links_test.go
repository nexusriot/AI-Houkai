package memory

import "testing"

func TestAddLinkIdempotent(t *testing.T) {
	m := &Memory{}
	addLink(m, "x", RelRelated)
	addLink(m, "x", RelRelated)
	if len(m.Links) != 1 {
		t.Errorf("addLink should be idempotent by (to, rel); got %d links", len(m.Links))
	}
}

func TestAddLinkDifferentRel(t *testing.T) {
	m := &Memory{}
	addLink(m, "x", RelRelated)
	addLink(m, "x", RelSupersedes)
	if len(m.Links) != 2 {
		t.Errorf("different rel to same target should add new link; got %d", len(m.Links))
	}
}

func TestRemoveLinksAllRels(t *testing.T) {
	m := &Memory{Links: []Link{
		{To: "x", Rel: RelRelated},
		{To: "x", Rel: RelSupersedes},
		{To: "y", Rel: RelRelated},
	}}
	removed := removeLinks(m, "x", "")
	if removed != 2 {
		t.Errorf("want 2 removed, got %d", removed)
	}
	if len(m.Links) != 1 || m.Links[0].To != "y" {
		t.Errorf("unexpected remaining links: %+v", m.Links)
	}
}

func TestRemoveLinksSpecificRel(t *testing.T) {
	m := &Memory{Links: []Link{
		{To: "x", Rel: RelRelated},
		{To: "x", Rel: RelSupersedes},
	}}
	removed := removeLinks(m, "x", RelRelated)
	if removed != 1 {
		t.Errorf("want 1 removed, got %d", removed)
	}
	if len(m.Links) != 1 || m.Links[0].Rel != RelSupersedes {
		t.Errorf("wrong link remaining: %+v", m.Links)
	}
}
