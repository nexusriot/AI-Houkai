package memory

import (
	"reflect"
	"testing"
)

func TestMetadataRoundTrip(t *testing.T) {
	original := Memory{
		ID:           "abc-123",
		Text:         "round-trip me",
		Type:         Semantic,
		Tags:         []string{"go", "test"},
		Importance:   0.75,
		CreatedAt:    1700000000,
		LastAccessed: 1700001000,
		AccessCount:  3,
		Source:       "unit-test",
		Links:        []Link{{To: "other", Rel: RelRelated}},
		SupersededBy: "next-id",
		SupersededAt: 1700002000,
		Polarity:     -1,
	}

	meta := MemoryToMetadata(original)
	round := MetadataToMemory(original.ID, original.Text, meta)

	if !reflect.DeepEqual(original.Tags, round.Tags) {
		t.Errorf("tags differ: got %v want %v", round.Tags, original.Tags)
	}
	if !reflect.DeepEqual(original.Links, round.Links) {
		t.Errorf("links differ: got %v want %v", round.Links, original.Links)
	}
	if round.Type != original.Type {
		t.Errorf("type: got %q want %q", round.Type, original.Type)
	}
	if round.Importance != original.Importance {
		t.Errorf("importance: got %.3f want %.3f", round.Importance, original.Importance)
	}
	if round.AccessCount != original.AccessCount {
		t.Errorf("access_count: got %d want %d", round.AccessCount, original.AccessCount)
	}
	if round.Source != original.Source {
		t.Errorf("source: got %q want %q", round.Source, original.Source)
	}
	if round.SupersededBy != original.SupersededBy {
		t.Errorf("superseded_by: got %q want %q", round.SupersededBy, original.SupersededBy)
	}
	if round.Polarity != original.Polarity {
		t.Errorf("polarity: got %d want %d", round.Polarity, original.Polarity)
	}
}

func TestMetadataEmptyTagsAndLinks(t *testing.T) {
	m := Memory{ID: "x", Type: Episodic, Tags: []string{}, Links: []Link{}}
	meta := MemoryToMetadata(m)
	round := MetadataToMemory("x", "", meta)
	if round.Tags == nil {
		t.Error("Tags should be non-nil empty slice after round-trip")
	}
	if len(round.Tags) != 0 {
		t.Errorf("Tags should be empty, got %v", round.Tags)
	}
	if round.Links == nil {
		t.Error("Links should be non-nil empty slice after round-trip")
	}
	if len(round.Links) != 0 {
		t.Errorf("Links should be empty, got %v", round.Links)
	}
}

func TestMetadataDefaultsTypeToEpisodic(t *testing.T) {
	m := MetadataToMemory("x", "y", map[string]string{})
	if m.Type != Episodic {
		t.Errorf("default type should be Episodic, got %q", m.Type)
	}
}

func TestConflictErrorMessage(t *testing.T) {
	e := &ConflictError{Conflicts: []Conflict{{Kind: KindDuplicate}, {Kind: KindContradiction}}}
	if e.Error() == "" {
		t.Error("ConflictError.Error() should not be empty")
	}
}
