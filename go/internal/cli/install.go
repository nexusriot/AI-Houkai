package cli

import (
	"fmt"

	"github.com/nexusriot/ai-houkai/internal/installer"
	"github.com/spf13/cobra"
)

// installFlags are shared by every `houkai install <client>` subcommand.
type installFlags struct {
	settingsPath string
	memPath      string
	collection   string
	binaryPath   string
	project      bool
	verify       bool
	print        bool
}

// target is the subset of every installer that `installFlags` overrides. The
// three installer types are distinct structs with identical fields, so the
// flag-application step reaches them through pointers rather than repeating
// the same six `if` blocks per client.
type target struct {
	memoryPath   *string
	collection   *string
	binaryPath   *string
	settingsPath *string
}

// apply overlays the user's flags onto an installer. projectPath is the
// client's project-scoped config location, used when --project is given.
func (f *installFlags) apply(t target, projectPath string) {
	if f.memPath != "" {
		*t.memoryPath = f.memPath
	}
	if f.collection != "" {
		*t.collection = f.collection
	}
	if f.binaryPath != "" {
		*t.binaryPath = f.binaryPath
	}
	if f.project {
		*t.settingsPath = projectPath
	}
	// --settings is the last word: it overrides --project.
	if f.settingsPath != "" {
		*t.settingsPath = f.settingsPath
	}
}

func (f *installFlags) register(cmd *cobra.Command) {
	cmd.Flags().StringVar(&f.settingsPath, "settings", "", "Explicit config file path (overrides --project)")
	cmd.Flags().StringVar(&f.memPath, "memory-path", "", "Memory store path")
	cmd.Flags().StringVar(&f.collection, "collection", "", "Collection name")
	cmd.Flags().StringVar(&f.binaryPath, "binary", "", "Path to ai-houkai-mcp binary")
	cmd.Flags().BoolVar(&f.project, "project", false, "Install project-scoped instead of globally")
	cmd.Flags().BoolVar(&f.verify, "verify", false, "Check binary + registration instead of installing")
	cmd.Flags().BoolVar(&f.print, "print", false, "Print the config block instead of writing it")
}

// clientInstaller is the slice of each installer that the install command
// drives. The three concrete types are unrelated structs with the same shape.
type clientInstaller interface {
	Install() (string, error)
	Verify() bool
	PrintConfig()
}

// installClient is everything that differs between `houkai install cursor`,
// `... opencode` and `... claude-code`. The command body itself — apply flags,
// then --<snippet> / --verify / --print / install — is one function below,
// because all three were the same 41 lines with three strings swapped.
type installClient struct {
	use         string
	short       string
	projectPath string
	// newInstaller returns a default installer — as a POINTER, since the
	// flags below mutate it through `target` after it is built — plus the
	// fields those flags write to.
	newInstaller func() (clientInstaller, target)
	// Optional "print the instruction-file snippet and exit" flag.
	snippetFlag string
	snippetHelp string
	snippetHead string
	snippet     func() string
	// afterInstall prints the client's own post-install guidance.
	afterInstall func()
}

func newInstallClientCmd(c installClient) *cobra.Command {
	var f installFlags
	var wantSnippet bool
	cmd := &cobra.Command{
		Use:   c.use,
		Short: c.short,
		RunE: func(cmd *cobra.Command, _ []string) error {
			inst, tgt := c.newInstaller()
			f.apply(tgt, c.projectPath)
			if wantSnippet {
				fmt.Println(c.snippetHead)
				fmt.Println()
				fmt.Println(c.snippet())
				return nil
			}
			if f.verify {
				return verifyInstall(*tgt.settingsPath, inst.Verify())
			}
			if f.print {
				inst.PrintConfig()
				return nil
			}
			path, err := inst.Install()
			if err != nil {
				return err
			}
			fmt.Printf("installed to %s\n", path)
			c.afterInstall()
			return nil
		},
	}
	f.register(cmd)
	if c.snippetFlag != "" {
		cmd.Flags().BoolVar(&wantSnippet, c.snippetFlag, false, c.snippetHelp)
	}
	return cmd
}

func newInstallCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "install",
		Short: "Register ai-houkai-mcp with an MCP client (claude-code, cursor, opencode)",
	}
	cmd.AddCommand(
		newInstallClaudeCodeCmd(),
		newInstallCursorCmd(),
		newInstallOpenCodeCmd(),
	)
	// Bare `houkai install` keeps its historical meaning: Claude Code.
	claudeFallback := newInstallClaudeCodeCmd()
	cmd.RunE = claudeFallback.RunE
	cmd.Flags().AddFlagSet(claudeFallback.Flags())
	return cmd
}

func newInstallClaudeCodeCmd() *cobra.Command {
	return newInstallClientCmd(installClient{
		use:         "claude-code",
		short:       "Register ai-houkai-mcp in Claude Code settings.json",
		projectPath: ".claude/settings.json",
		newInstaller: func() (clientInstaller, target) {
			i := installer.DefaultInstaller()
			return &i, target{&i.MemoryPath, &i.Collection, &i.BinaryPath, &i.SettingsPath}
		},
		afterInstall: func() { fmt.Println(installer.ClaudeMDSnippet()) },
	})
}

func newInstallCursorCmd() *cobra.Command {
	return newInstallClientCmd(installClient{
		use:         "cursor",
		short:       "Register ai-houkai-mcp in Cursor's mcp.json",
		projectPath: installer.CursorProjectConfigPath,
		newInstaller: func() (clientInstaller, target) {
			i := installer.DefaultCursorInstaller()
			return &i, target{&i.MemoryPath, &i.Collection, &i.BinaryPath, &i.SettingsPath}
		},
		snippetFlag: "rule",
		snippetHelp: "Print a .cursor/rules/*.mdc memory-usage snippet",
		snippetHead: ".cursor/rules/ai-houkai-memory.mdc",
		snippet:     func() string { return installer.CursorRuleSnippet },
		afterInstall: func() {
			fmt.Println("Reload Cursor, then check Settings → MCP.")
			fmt.Println("Tip: `houkai install cursor --rule` prints a .cursor/rules memory-usage snippet.")
		},
	})
}

func newInstallOpenCodeCmd() *cobra.Command {
	return newInstallClientCmd(installClient{
		use:         "opencode",
		short:       "Register ai-houkai-mcp in OpenCode's opencode.json",
		projectPath: installer.OpenCodeProjectConfigPath,
		newInstaller: func() (clientInstaller, target) {
			i := installer.DefaultOpenCodeInstaller()
			return &i, target{&i.MemoryPath, &i.Collection, &i.BinaryPath, &i.SettingsPath}
		},
		snippetFlag: "agents",
		snippetHelp: "Print an AGENTS.md memory-usage snippet",
		snippetHead: "AGENTS.md snippet",
		snippet:     func() string { return installer.OpenCodeAgentsSnippet },
		afterInstall: func() {
			fmt.Println("Restart OpenCode to load the memory tools.")
			fmt.Println("Tip: `houkai install opencode --agents` prints an AGENTS.md memory-usage snippet.")
		},
	})
}

// verifyInstall reports the smoke-test result for an installer target.
func verifyInstall(settingsPath string, registered bool) error {
	cmd, ok := installer.VerifyBinary()
	if ok {
		fmt.Printf("  ok   mcp binary: %s\n", cmd)
	} else {
		fmt.Printf("  err  %q not on PATH — install the ai-houkai-mcp binary\n", cmd)
	}
	if registered {
		fmt.Printf("  ok   registered in %s\n", settingsPath)
	} else {
		fmt.Printf("  warn not yet in %s — run install first\n", settingsPath)
	}
	if !ok || !registered {
		return fmt.Errorf("verification failed")
	}
	return nil
}
