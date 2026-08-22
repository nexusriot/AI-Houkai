//go:build !unix

package memory

// lockTrash degrades to a no-op where flock(2) is unavailable; see the unix
// implementation for what it protects.
func (s *MemoryStore) lockTrash() func() {
	return func() {}
}
