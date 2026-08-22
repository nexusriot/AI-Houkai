//go:build !unix

package memory

// lockTrash serializes trash mutations within the process via the shared
// mutex; without flock(2) there is no cross-process exclusion here, so
// sharing one trash file between PROCESSES is unsynchronized on this
// platform (a documented limitation — in-process multi-store use is covered).
func (s *MemoryStore) lockTrash() (func(), error) {
	trashMu.Lock()
	return trashMu.Unlock, nil
}
