package memory

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

type MemoryType string

const (
	Episodic   MemoryType = "episodic"
	Semantic   MemoryType = "semantic"
	Procedural MemoryType = "procedural"
	Feedback   MemoryType = "feedback"
)

type Link struct {
	To  string `json:"to"`
	Rel string `json:"rel"`
}

type Memory struct {
	ID           string     `json:"id"`
	Text         string     `json:"text"`
	Type         MemoryType `json:"type"`
	Tags         []string   `json:"tags"`
	Importance   float32    `json:"importance"`
	CreatedAt    float64    `json:"created_at"`
	LastAccessed float64    `json:"last_accessed"`
	AccessCount  int        `json:"access_count"`
	Source       string     `json:"source,omitempty"`
	Links        []Link     `json:"links"`
	SupersededBy string     `json:"superseded_by"`
	SupersededAt float64    `json:"superseded_at"`
	Polarity     int        `json:"polarity"`
}

type MemoryWithScore struct {
	Memory
	Score float32 `json:"score"`
}

type HybridWeights struct {
	Cosine     float32
	Lexical    float32
	Recency    float32
	Importance float32
}

func DefaultWeights() HybridWeights {
	return HybridWeights{Cosine: 0.55, Lexical: 0.20, Recency: 0.15, Importance: 0.10}
}

type ExpandSpec struct {
	Rels  []string
	Depth int
	Cap   int
	Score float32
}

type ConflictKind string

const (
	KindContradiction ConflictKind = "contradiction"
	KindDuplicate     ConflictKind = "duplicate"
)

type Conflict struct {
	Kind       ConflictKind `json:"kind"`
	Reason     string       `json:"reason"`
	Similarity float32      `json:"similarity"`
	A          Memory       `json:"a"`
	B          Memory       `json:"b"`
}

type ConflictError struct {
	Conflicts []Conflict
}

func (e *ConflictError) Error() string {
	return fmt.Sprintf("conflict detected: %d conflict(s)", len(e.Conflicts))
}

type Graph struct {
	Nodes []Memory
	Edges []struct {
		From string
		To   string
		Rel  string
	}
}

type ConflictPolicy string

const (
	PolicyIgnore    ConflictPolicy = "ignore"
	PolicyWarn      ConflictPolicy = "warn"
	PolicySupersede ConflictPolicy = "supersede"
	PolicyRaise     ConflictPolicy = "raise"
)

// MetadataToMemory reconstructs a Memory from chromem-go metadata + id + text.
func MetadataToMemory(id, text string, meta map[string]string) Memory {
	m := Memory{
		ID:   id,
		Text: text,
		Type: Episodic,
	}
	if v, ok := meta["type"]; ok {
		m.Type = MemoryType(v)
	}
	if v, ok := meta["tags"]; ok && v != "" {
		m.Tags = strings.Split(v, ",")
	} else {
		m.Tags = []string{}
	}
	if v, ok := meta["importance"]; ok {
		f, _ := strconv.ParseFloat(v, 32)
		m.Importance = float32(f)
	}
	if v, ok := meta["created_at"]; ok {
		m.CreatedAt, _ = strconv.ParseFloat(v, 64)
	}
	if v, ok := meta["last_accessed"]; ok {
		m.LastAccessed, _ = strconv.ParseFloat(v, 64)
	}
	if v, ok := meta["access_count"]; ok {
		n, _ := strconv.Atoi(v)
		m.AccessCount = n
	}
	if v, ok := meta["source"]; ok {
		m.Source = v
	}
	if v, ok := meta["links"]; ok && v != "" && v != "[]" {
		_ = json.Unmarshal([]byte(v), &m.Links)
	}
	if m.Links == nil {
		m.Links = []Link{}
	}
	if v, ok := meta["superseded_by"]; ok {
		m.SupersededBy = v
	}
	if v, ok := meta["superseded_at"]; ok {
		m.SupersededAt, _ = strconv.ParseFloat(v, 64)
	}
	if v, ok := meta["polarity"]; ok {
		m.Polarity, _ = strconv.Atoi(v)
	}
	return m
}

// MemoryToMetadata serialises a Memory to chromem-go metadata (all string values).
func MemoryToMetadata(m Memory) map[string]string {
	linksJSON, _ := json.Marshal(m.Links)
	tags := ""
	if len(m.Tags) > 0 {
		tags = strings.Join(m.Tags, ",")
	}
	return map[string]string{
		"type":          string(m.Type),
		"tags":          tags,
		"importance":    fmt.Sprintf("%f", m.Importance),
		"created_at":    fmt.Sprintf("%f", m.CreatedAt),
		"last_accessed": fmt.Sprintf("%f", m.LastAccessed),
		"access_count":  strconv.Itoa(m.AccessCount),
		"source":        m.Source,
		"links":         string(linksJSON),
		"superseded_by": m.SupersededBy,
		"superseded_at": fmt.Sprintf("%f", m.SupersededAt),
		"polarity":      strconv.Itoa(m.Polarity),
	}
}

var ErrNotFound = errors.New("memory not found")
