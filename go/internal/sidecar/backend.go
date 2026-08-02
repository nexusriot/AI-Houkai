package sidecar

import (
	"context"

	"github.com/nexusriot/ai-houkai/internal/vector"
)

// RowBuilder turns a backend Item into the flat row the index stores. The
// store supplies it so this package never imports memory (a cycle).
type RowBuilder func(vector.Item) Row

// IndexedBackend is a write-through proxy over a vector.Backend that mirrors
// every mutation into a sidecar Index.
//
// Wrapping the backend rather than patching call sites is deliberate: the
// store writes through Add / UpdateMetadata / Delete from many places, and an
// index that misses one is worse than no index at all. Reads pass straight
// through untouched.
type IndexedBackend struct {
	inner vector.Backend
	index *Index
	row   RowBuilder
}

// Wrap returns b decorated so writes also land in idx.
func Wrap(b vector.Backend, idx *Index, row RowBuilder) *IndexedBackend {
	return &IndexedBackend{inner: b, index: idx, row: row}
}

// Inner exposes the undecorated backend, for a reindex that must read the
// authoritative rows without recursing back through the mirror.
func (d *IndexedBackend) Inner() vector.Backend { return d.inner }

// Index exposes the sidecar so the store can query it.
func (d *IndexedBackend) Index() *Index { return d.index }

func (d *IndexedBackend) Add(ctx context.Context, items []vector.Item) error {
	if err := d.inner.Add(ctx, items); err != nil {
		return err
	}
	if d.index.Healthy() && len(items) > 0 {
		rows := make([]Row, len(items))
		for i, it := range items {
			rows[i] = d.row(it)
		}
		d.index.Upsert(rows)
	}
	return nil
}

func (d *IndexedBackend) UpdateMetadata(ctx context.Context, id string, meta map[string]string) error {
	if err := d.inner.UpdateMetadata(ctx, id, meta); err != nil {
		return err
	}
	if !d.index.Healthy() {
		return nil
	}
	// A metadata-only update carries no text, so re-read the canonical item
	// rather than indexing a partial row.
	items, err := d.inner.Get(ctx, []string{id})
	if err != nil {
		d.index.Disable("read-back failed: " + err.Error())
		return nil
	}
	rows := make([]Row, 0, len(items))
	for _, it := range items {
		rows = append(rows, d.row(it))
	}
	d.index.Upsert(rows)
	return nil
}

func (d *IndexedBackend) Delete(ctx context.Context, ids []string) error {
	if err := d.inner.Delete(ctx, ids); err != nil {
		return err
	}
	d.index.Delete(ids)
	return nil
}

func (d *IndexedBackend) Query(ctx context.Context, embedding []float32, k int) ([]vector.Hit, error) {
	return d.inner.Query(ctx, embedding, k)
}

func (d *IndexedBackend) Get(ctx context.Context, ids []string) ([]vector.Item, error) {
	return d.inner.Get(ctx, ids)
}

func (d *IndexedBackend) All(ctx context.Context) ([]vector.Item, error) {
	return d.inner.All(ctx)
}

func (d *IndexedBackend) Count(ctx context.Context) (int, error) {
	return d.inner.Count(ctx)
}

func (d *IndexedBackend) Close() error {
	// Close the index first: it is derived, so losing it is recoverable, but
	// leaving its file handle open after the backend is gone is not tidy.
	_ = d.index.Close()
	return d.inner.Close()
}
