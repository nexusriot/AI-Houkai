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

func (f *installFlags) register(cmd *cobra.Command) {
	cmd.Flags().StringVar(&f.settingsPath, "settings", "", "Explicit config file path (overrides --project)")
	cmd.Flags().StringVar(&f.memPath, "memory-path", "", "Memory store path")
	cmd.Flags().StringVar(&f.collection, "collection", "", "Collection name")
	cmd.Flags().StringVar(&f.binaryPath, "binary", "", "Path to ai-houkai-mcp binary")
	cmd.Flags().BoolVar(&f.project, "project", false, "Install project-scoped instead of globally")
	cmd.Flags().BoolVar(&f.verify, "verify", false, "Check binary + registration instead of installing")
	cmd.Flags().BoolVar(&f.print, "print", false, "Print the config block instead of writing it")
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
	var f installFlags
	cmd := &cobra.Command{
		Use:   "claude-code",
		Short: "Register ai-houkai-mcp in Claude Code settings.json",
		RunE: func(cmd *cobra.Command, _ []string) error {
			inst := installer.DefaultInstaller()
			if f.memPath != "" {
				inst.MemoryPath = f.memPath
			}
			if f.collection != "" {
				inst.Collection = f.collection
			}
			if f.binaryPath != "" {
				inst.BinaryPath = f.binaryPath
			}
			if f.project {
				inst.SettingsPath = ".claude/settings.json"
			}
			if f.settingsPath != "" {
				inst.SettingsPath = f.settingsPath
			}
			if f.verify {
				return verifyInstall(inst.SettingsPath, inst.Verify())
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
			fmt.Println(installer.ClaudeMDSnippet())
			return nil
		},
	}
	f.register(cmd)
	return cmd
}

func newInstallCursorCmd() *cobra.Command {
	var f installFlags
	var rule bool
	cmd := &cobra.Command{
		Use:   "cursor",
		Short: "Register ai-houkai-mcp in Cursor's mcp.json",
		RunE: func(cmd *cobra.Command, _ []string) error {
			inst := installer.DefaultCursorInstaller()
			if f.memPath != "" {
				inst.MemoryPath = f.memPath
			}
			if f.collection != "" {
				inst.Collection = f.collection
			}
			if f.binaryPath != "" {
				inst.BinaryPath = f.binaryPath
			}
			if f.project {
				inst.SettingsPath = installer.CursorProjectConfigPath
			}
			if f.settingsPath != "" {
				inst.SettingsPath = f.settingsPath
			}
			if rule {
				fmt.Println(".cursor/rules/ai-houkai-memory.mdc")
				fmt.Println()
				fmt.Println(installer.CursorRuleSnippet)
				return nil
			}
			if f.verify {
				return verifyInstall(inst.SettingsPath, inst.Verify())
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
			fmt.Println("Reload Cursor, then check Settings → MCP.")
			fmt.Println("Tip: `houkai install cursor --rule` prints a .cursor/rules memory-usage snippet.")
			return nil
		},
	}
	f.register(cmd)
	cmd.Flags().BoolVar(&rule, "rule", false, "Print a .cursor/rules/*.mdc memory-usage snippet")
	return cmd
}

func newInstallOpenCodeCmd() *cobra.Command {
	var f installFlags
	var agents bool
	cmd := &cobra.Command{
		Use:   "opencode",
		Short: "Register ai-houkai-mcp in OpenCode's opencode.json",
		RunE: func(cmd *cobra.Command, _ []string) error {
			inst := installer.DefaultOpenCodeInstaller()
			if f.memPath != "" {
				inst.MemoryPath = f.memPath
			}
			if f.collection != "" {
				inst.Collection = f.collection
			}
			if f.binaryPath != "" {
				inst.BinaryPath = f.binaryPath
			}
			if f.project {
				inst.SettingsPath = installer.OpenCodeProjectConfigPath
			}
			if f.settingsPath != "" {
				inst.SettingsPath = f.settingsPath
			}
			if agents {
				fmt.Println("AGENTS.md snippet")
				fmt.Println()
				fmt.Println(installer.OpenCodeAgentsSnippet)
				return nil
			}
			if f.verify {
				return verifyInstall(inst.SettingsPath, inst.Verify())
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
			fmt.Println("Restart OpenCode to load the memory tools.")
			fmt.Println("Tip: `houkai install opencode --agents` prints an AGENTS.md memory-usage snippet.")
			return nil
		},
	}
	f.register(cmd)
	cmd.Flags().BoolVar(&agents, "agents", false, "Print an AGENTS.md memory-usage snippet")
	return cmd
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
