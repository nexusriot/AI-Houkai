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
	// ExpiresAt is the Unix timestamp after which the memory is treated as
	// expired: hidden from recall/list and reclaimable by PurgeExpired.
	// 0 means "never expires".
	ExpiresAt float64 `json:"expires_at"`
	// Pinned marks a standing instruction: always offered to the packer and
	// never pruned by decay. Importance cannot express this — it drives
	// ranking, decay survival and the MinImportance filter at once, so raising
	// it to protect an instruction also distorts every search.
	Pinned bool `json:"pinned"`
	// Trust records how much the memory's ORIGIN is trusted, distinct from how
	// important or how confident it is. Anything reaching Remember becomes
	// durable, well-ranked agent context later, so a fact scraped from a page
	// and one stated by the user need to be distinguishable at recall time.
	// Empty reads as TrustTrusted so existing stores are unchanged.
	Trust TrustLevel `json:"trust,omitempty"`
	// ContentHash is the hash of the normalised text, set when Remember is
	// called with Idempotent. Lets a repeated assertion be recognised without
	// a vector query.
	ContentHash string `json:"content_hash,omitempty"`
}

// TrustLevel is how much the ORIGIN of a memory is trusted:
//
//	trusted   — stated by the principal (the user, or the operator's own config)
//	reported  — relayed by a tool or another agent; plausible, unverified
//	untrusted — from content the agent merely read (a page, a document, an
//	            email); treat as data, never as instruction
//
// Best-effort labelling, not a security boundary.
type TrustLevel string

const (
	TrustTrusted   TrustLevel = "trusted"
	TrustReported  TrustLevel = "reported"
	TrustUntrusted TrustLevel = "untrusted"
)

// TrustLevels is the validated vocabulary, ordered most- to least-trusted so
// an index comparison implements "at least this trusted".
var TrustLevels = []string{"trusted", "reported", "untrusted"}

// TrustRank returns the position of a level in TrustLevels; an empty or
// unknown value reads as trusted, matching how old rows deserialise.
func TrustRank(t TrustLevel) int {
	for i, s := range TrustLevels {
		if string(t) == s {
			return i
		}
	}
	return 0
}

type MemoryWithScore struct {
	Memory
	Score float32 `json:"score"`
	// Explain, when non-nil, carries the per-signal score breakdown produced
	// by Recall(..., Explain=true). Omitted from JSON otherwise.
	Explain map[string]any `json:"explain,omitempty"`
}

// FusionMode selects how the hybrid signals are combined.
type FusionMode string

const (
	FusionWeighted FusionMode = "weighted"
	FusionRRF      FusionMode = "rrf"
)

type HybridWeights struct {
	Cosine     float32
	Lexical    float32
	Recency    float32
	Importance float32
	// PolarityWeight is an additive bonus of +weight for polarity=+1 and
	// -weight for polarity=-1 (0 disables the nudge).
	PolarityWeight float32
	// Graph is the graph-proximity weight: a candidate's connectedness (via
	// links) to the other strong hits in the pool lifts its score — a
	// lightweight HippoRAG-style associative signal. Additive on top of the
	// core signals like PolarityWeight; 0 (default) disables the graph channel
	// so scoring is byte-identical to before. See graphSpread.
	Graph float32
	// RecencyBasis selects which timestamp the recency term measures:
	// "created" (default) scores by how recently the fact was learned — stable
	// across recalls; "accessed" scores by how recently it was retrieved
	// (self-reinforcing, the old behaviour). The empty string means "created".
	RecencyBasis string
}

func DefaultWeights() HybridWeights {
	return HybridWeights{
		Cosine: 0.55, Lexical: 0.20, Recency: 0.15, Importance: 0.10,
		PolarityWeight: 0.05, RecencyBasis: "created",
	}
}

type ExpandSpec struct {
	Rels  []string
	Depth int
	Cap   int
	Score float32
	// Decay is the per-hop score multiplier beyond the first hop: a hop-h
	// neighbour is scored Score*Decay^(h-1). 0 or 1 keeps every expanded node
	// at Score regardless of distance (backward-compatible).
	Decay float32
	// Rerank, when true, merges expanded neighbours into the candidate pool
	// before dedup, MMR diversity and the top-k cut, so they compete for the k
	// slots and can neither inject near-duplicates nor overflow k. The
	// min_cosine gate does not apply to them — they are graph-justified, not
	// cosine-justified. When false (default) they are appended AFTER the top-k
	// cut, unfiltered — the original, backward-compatible behaviour.
	Rerank bool
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
		Type: Semantic, // default when metadata carries no type (matches Python)
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
	// Old rows written before TTL landed have no "expires_at" key; the
	// zero-value 0 = never expires, so they keep showing up in recall.
	if v, ok := meta["expires_at"]; ok {
		m.ExpiresAt, _ = strconv.ParseFloat(v, 64)
	}
	// Likewise for the fields added after TTL: an absent key must read as the
	// neutral value, or an old store would change behaviour just by being
	// opened with a newer build.
	if v, ok := meta["pinned"]; ok {
		m.Pinned = v == "true"
	}
	if v, ok := meta["trust"]; ok && v != "" {
		m.Trust = TrustLevel(v)
	} else {
		m.Trust = TrustTrusted
	}
	if v, ok := meta["content_hash"]; ok {
		m.ContentHash = v
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
		"expires_at":    fmt.Sprintf("%f", m.ExpiresAt),
		"pinned":        strconv.FormatBool(m.Pinned),
		"trust":         string(trustOrDefault(m.Trust)),
		"content_hash":  m.ContentHash,
	}
}

// trustOrDefault normalises the zero value to "trusted".
func trustOrDefault(t TrustLevel) TrustLevel {
	if t == "" {
		return TrustTrusted
	}
	return t
}

var ErrNotFound = errors.New("memory not found")

// ToDict produces a full, self-contained snapshot of the memory — used by
// the audit journal and by the .ahkai export format.
func (m Memory) ToDict() map[string]any {
	tags := make([]any, len(m.Tags))
	for i, t := range m.Tags {
		tags[i] = t
	}
	links := make([]any, len(m.Links))
	for i, l := range m.Links {
		links[i] = map[string]any{"to": l.To, "rel": l.Rel}
	}
	return map[string]any{
		"id":            m.ID,
		"text":          m.Text,
		"type":          string(m.Type),
		"tags":          tags,
		"importance":    m.Importance,
		"created_at":    m.CreatedAt,
		"last_accessed": m.LastAccessed,
		"access_count":  m.AccessCount,
		"source":        m.Source,
		"links":         links,
		"superseded_by": m.SupersededBy,
		"superseded_at": m.SupersededAt,
		"polarity":      m.Polarity,
		"expires_at":    m.ExpiresAt,
		"pinned":        m.Pinned,
		"trust":         string(trustOrDefault(m.Trust)),
		"content_hash":  m.ContentHash,
	}
}

// MemoryFromDict rehydrates a Memory from a ToDict() / journal payload.
func MemoryFromDict(d map[string]any) Memory {
	asString := func(v any) string {
		s, _ := v.(string)
		return s
	}
	asFloat := func(v any) float64 {
		switch x := v.(type) {
		case float64:
			return x
		case float32:
			return float64(x)
		case int:
			return float64(x)
		case int64:
			return float64(x)
		}
		return 0
	}
	asInt := func(v any) int {
		switch x := v.(type) {
		case float64:
			return int(x)
		case int:
			return x
		case int64:
			return int(x)
		}
		return 0
	}
	tags := []string{}
	if raw, ok := d["tags"].([]any); ok {
		for _, t := range raw {
			if s, ok := t.(string); ok {
				tags = append(tags, s)
			}
		}
	}
	links := []Link{}
	if raw, ok := d["links"].([]any); ok {
		for _, l := range raw {
			lm, _ := l.(map[string]any)
			links = append(links, Link{To: asString(lm["to"]), Rel: asString(lm["rel"])})
		}
	}
	mt := MemoryType(asString(d["type"]))
	if mt == "" {
		mt = Semantic
	}
	return Memory{
		ID:           asString(d["id"]),
		Text:         asString(d["text"]),
		Type:         mt,
		Tags:         tags,
		Importance:   float32(asFloat(d["importance"])),
		CreatedAt:    asFloat(d["created_at"]),
		LastAccessed: asFloat(d["last_accessed"]),
		AccessCount:  asInt(d["access_count"]),
		Source:       asString(d["source"]),
		Links:        links,
		SupersededBy: asString(d["superseded_by"]),
		SupersededAt: asFloat(d["superseded_at"]),
		Polarity:     asInt(d["polarity"]),
		ExpiresAt:    asFloat(d["expires_at"]),
		Pinned:       asBool(d["pinned"]),
		Trust:        trustOrDefault(TrustLevel(asString(d["trust"]))),
		ContentHash:  asString(d["content_hash"]),
	}
}

func asBool(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case string:
		return x == "true"
	}
	return false
}
