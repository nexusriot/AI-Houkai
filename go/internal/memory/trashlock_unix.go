//go:build unix

package memory

import (
	"fmt"
	"os"
	"syscall"
)

// lockTrash serializes trash mutations and returns the release func. The
// trash is a shared read-modify-write file: two stores (one per collection on
// the same path — a supported layout, or two processes) mutating it
// concurrently would lose whichever rewrite lands first.
//
// A process-wide mutex is the floor on every platform; on unix an exclusive
// flock on a lock file beside the trash extends the exclusion across
// processes. Failure to take the flock is an error, not a silent downgrade —
// proceeding unlocked risks exactly the lost-rewrite corruption the lock
// exists to prevent (and a directory where the lock file cannot be created
// cannot hold the trash file either).
func (s *MemoryStore) lockTrash() (func(), error) {
	trashMu.Lock()
	f, err := os.OpenFile(s.TrashPath()+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		trashMu.Unlock()
		return nil, fmt.Errorf("trash lock: %w", err)
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX); err != nil {
		_ = f.Close()
		trashMu.Unlock()
		return nil, fmt.Errorf("trash lock: %w", err)
	}
	return func() {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		_ = f.Close()
		trashMu.Unlock()
	}, nil
}
