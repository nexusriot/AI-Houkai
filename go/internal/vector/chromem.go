package vector

import (
	"context"
	"fmt"
	"math"

	chromem "github.com/philippgille/chromem-go"
)

// ChromemBackend wraps philippgille/chromem-go.
type ChromemBackend struct {
	db         *chromem.DB
	collection *chromem.Collection
	dim        int
}

func NewChromem(path, collection string, dim int) (*ChromemBackend, error) {
	db, err := chromem.NewPersistentDB(path, false)
	if err != nil {
		return nil, fmt.Errorf("chromem open %s: %w", path, err)
	}
	// Use a no-op embedding function — we always supply pre-computed vectors.
	col, err := db.GetOrCreateCollection(collection, map[string]string{}, noopEmbedFn)
	if err != nil {
		return nil, fmt.Errorf("chromem collection: %w", err)
	}
	return &ChromemBackend{db: db, collection: col, dim: dim}, nil
}

// noopEmbedFn is used when all embeddings are supplied by the caller.
var noopEmbedFn chromem.EmbeddingFunc = func(ctx context.Context, text string) ([]float32, error) {
	return nil, fmt.Errorf("no embedding function configured; provide pre-computed vectors")
}

func (b *ChromemBackend) Add(ctx context.Context, items []Item) error {
	docs := make([]chromem.Document, len(items))
	for i, it := range items {
		docs[i] = chromem.Document{
			ID:        it.ID,
			Content:   it.Content,
			Embedding: it.Embedding,
			Metadata:  it.Metadata,
		}
	}
	return b.collection.AddDocuments(ctx, docs, 1)
}

func (b *ChromemBackend) Query(ctx context.Context, embedding []float32, k int) ([]Hit, error) {
	// chromem-go errors out if k > Count(). Clamp to whichever is smaller.
	n := b.collection.Count()
	if n == 0 {
		return nil, nil
	}
	if k > n {
		k = n
	}
	results, err := b.collection.QueryEmbedding(ctx, embedding, k, nil, nil)
	if err != nil {
		return nil, err
	}
	hits := make([]Hit, len(results))
	for i, r := range results {
		hits[i] = Hit{
			Item: Item{
				ID:        r.ID,
				Content:   r.Content,
				Embedding: r.Embedding,
				Metadata:  r.Metadata,
			},
			Similarity: r.Similarity,
		}
	}
	return hits, nil
}

func (b *ChromemBackend) Get(ctx context.Context, ids []string) ([]Item, error) {
	out := make([]Item, 0, len(ids))
	for _, id := range ids {
		doc, err := b.collection.GetByID(ctx, id)
		if err != nil {
			continue // skip missing
		}
		out = append(out, Item{
			ID:        doc.ID,
			Content:   doc.Content,
			Embedding: doc.Embedding,
			Metadata:  doc.Metadata,
		})
	}
	return out, nil
}

func (b *ChromemBackend) All(ctx context.Context) ([]Item, error) {
	n := b.collection.Count()
	if n == 0 {
		return nil, nil
	}
	// Use a zero vector as query anchor to retrieve all documents.
	zeroVec := make([]float32, b.dim)
	results, err := b.collection.QueryEmbedding(ctx, zeroVec, n, nil, nil)
	if err != nil {
		return nil, err
	}
	items := make([]Item, len(results))
	for i, r := range results {
		items[i] = Item{
			ID:        r.ID,
			Content:   r.Content,
			Embedding: r.Embedding,
			Metadata:  r.Metadata,
		}
	}
	return items, nil
}

func (b *ChromemBackend) UpdateMetadata(ctx context.Context, id string, meta map[string]string) error {
	doc, err := b.collection.GetByID(ctx, id)
	if err != nil {
		return fmt.Errorf("UpdateMetadata get %s: %w", id, err)
	}
	for k, v := range meta {
		doc.Metadata[k] = v
	}
	_ = b.collection.Delete(ctx, nil, nil, id)
	return b.collection.AddDocuments(ctx, []chromem.Document{doc}, 1)
}

func (b *ChromemBackend) Delete(ctx context.Context, ids []string) error {
	return b.collection.Delete(ctx, nil, nil, ids...)
}

func (b *ChromemBackend) Count(_ context.Context) (int, error) {
	return b.collection.Count(), nil
}

func (b *ChromemBackend) Close() error {
	return nil // chromem-go PersistentDB flushes on each write
}

// ListCollections returns every collection name with its document count.
func (b *ChromemBackend) ListCollections() map[string]int {
	out := map[string]int{}
	for name, col := range b.db.ListCollections() {
		out[name] = col.Count()
	}
	return out
}

// HasCollection reports whether a collection exists.
func (b *ChromemBackend) HasCollection(name string) bool {
	_, ok := b.db.ListCollections()[name]
	return ok
}

// CreateCollection creates an empty collection.
func (b *ChromemBackend) CreateCollection(name string) error {
	_, err := b.db.CreateCollection(name, map[string]string{}, noopEmbedFn)
	return err
}

// DeleteCollection removes a collection and every document in it.
func (b *ChromemBackend) DeleteCollection(name string) error {
	return b.db.DeleteCollection(name)
}

// CopyCollection copies all documents (embeddings included — no
// re-embedding) from src to dst. dst is created if missing; existing dst
// ids are overwritten. Returns the number of copied documents.
func (b *ChromemBackend) CopyCollection(ctx context.Context, src, dst string) (int, error) {
	srcCol := b.db.GetCollection(src, noopEmbedFn)
	if srcCol == nil {
		return 0, fmt.Errorf("collection %q not found", src)
	}
	n := srcCol.Count()
	if n == 0 {
		return 0, nil
	}
	// Retrieve every document via the zero-vector query trick (see All).
	zeroVec := make([]float32, b.dim)
	results, err := srcCol.QueryEmbedding(ctx, zeroVec, n, nil, nil)
	if err != nil {
		return 0, err
	}
	dstCol, err := b.db.GetOrCreateCollection(dst, map[string]string{}, noopEmbedFn)
	if err != nil {
		return 0, err
	}
	docs := make([]chromem.Document, len(results))
	for i, r := range results {
		docs[i] = chromem.Document{
			ID:        r.ID,
			Content:   r.Content,
			Embedding: r.Embedding,
			Metadata:  r.Metadata,
		}
	}
	// AddDocuments upserts, so existing dst ids are overwritten.
	if err := dstCol.AddDocuments(ctx, docs, 1); err != nil {
		return 0, err
	}
	return len(docs), nil
}

// cosine similarity utility (also used by reflection engine via package access).
func CosineSim(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
