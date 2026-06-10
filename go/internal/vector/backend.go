package vector

import "context"

// Item is a unit stored in the vector backend.
type Item struct {
	ID        string
	Content   string
	Embedding []float32
	Metadata  map[string]string
}

// Hit is a query result from the vector backend.
type Hit struct {
	Item
	Similarity float32
}

// Backend abstracts the vector database.
type Backend interface {
	// Add upserts one or more items.
	Add(ctx context.Context, items []Item) error
	// Query returns up to k nearest neighbours to the given embedding.
	Query(ctx context.Context, embedding []float32, k int) ([]Hit, error)
	// Get fetches items by ID. Missing IDs are silently omitted.
	Get(ctx context.Context, ids []string) ([]Item, error)
	// All returns every item (used by ReflectionEngine for clustering).
	All(ctx context.Context) ([]Item, error)
	// UpdateMetadata patches the metadata of an existing item.
	UpdateMetadata(ctx context.Context, id string, meta map[string]string) error
	// Delete removes items by ID.
	Delete(ctx context.Context, ids []string) error
	// Count returns the total number of stored items.
	Count(ctx context.Context) (int, error)
	// Close flushes / releases resources.
	Close() error
}
