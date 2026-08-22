//go:build unix

package memory

import (
	"os"
	"syscall"
)

// lockTrash takes an exclusive flock on a lock file beside the trash file and
// returns the release func. The trash is a shared read-modify-write file: two
// stores (one per collection on the same path — a supported layout, or two
// processes) mutating it concurrently would lose whichever rewrite lands
// first. flock excludes between separate fds, so it covers goroutines within
// one process and other processes alike. Errors degrade to no locking rather
// than failing the operation — the lock is protection, not a prerequisite.
func (s *MemoryStore) lockTrash() func() {
	f, err := os.OpenFile(s.TrashPath()+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return func() {}
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX); err != nil {
		_ = f.Close()
		return func() {}
	}
	return func() {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		_ = f.Close()
	}
}
