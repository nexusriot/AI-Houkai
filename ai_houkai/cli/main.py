"""houkai — AI-Houkai CLI entrypoint."""

from __future__ import annotations

import os
from typing import Optional
import typer

from importlib.metadata import version, PackageNotFoundError

from ai_houkai.cli.config import load as load_config
from ai_houkai.memory_system import MemoryStore
from ai_houkai.cli.commands.remember import remember
from ai_houkai.cli.commands.recall import recall
from ai_houkai.cli.commands.pack import pack, auto_context
from ai_houkai.cli.commands.list_cmd import list_memories
from ai_houkai.cli.commands.show import show
from ai_houkai.cli.commands.forget import forget
from ai_houkai.cli.commands.nuke import nuke
from ai_houkai.cli.commands.edit import edit, tag, bump
from ai_houkai.cli.commands.link import link, unlink, neighbors, graph
from ai_houkai.cli.commands.conflicts import conflicts, supersede, restore
from ai_houkai.cli.commands.curation import (
    merge,
    path,
    tags_app,
    trash_app,
    versions,
)
from ai_houkai.cli.commands.decay import prune, purge
from ai_houkai.cli.commands.reflect import reflect
from ai_houkai.cli.commands.io import export_cmd, import_cmd, info_cmd, backup
from ai_houkai.cli.commands.stats import stats
from ai_houkai.cli.commands.doctor import doctor
from ai_houkai.cli.commands.eval_cmd import eval_cmd
from ai_houkai.cli.commands.ingest import ingest
from ai_houkai.cli.commands.serve import serve
from ai_houkai.cli.commands.collections import collections_app
from ai_houkai.cli.commands.tui_cmd import tui
from ai_houkai.cli.commands.maintenance import maintenance_app
from ai_houkai.cli.commands.journal import journal_app
from ai_houkai.cli.commands.timetravel import (
    get_at,
    history,
    metrics,
    state_at,
)


app = typer.Typer(
    name="houkai",
    help="Manage AI-Houkai memories from the terminal.",
    no_args_is_help=True,
    add_completion=True,
)


def _version_callback(value: bool) -> None:
    if not value:
        return
    try:
        v = version("ai-houkai")
    except PackageNotFoundError:
        v = "unknown (not installed as a distribution)"
    typer.echo(f"houkai {v}")
    raise typer.Exit()


@app.callback()
def _callback(
    ctx: typer.Context,
    store_path: Optional[str] = typer.Option(
        None, "--store", "-S", envvar="AI_HOUKAI_PATH",
        help="Override Chroma store path",
    ),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-C", envvar="AI_HOUKAI_COLLECTION",
        help="Override collection name",
    ),
    _version: Optional[bool] = typer.Option(
        None, "--version", "-V",
        callback=_version_callback, is_eager=True, expose_value=False,
        help="Show the installed ai-houkai version and exit.",
    ),
) -> None:
    cfg = load_config()
    if store_path:
        # A quoted/env-provided `~/…` reaches us unexpanded — expand it here
        # or Chroma creates a literal ./~ directory.
        cfg.store_path = os.path.expanduser(store_path)
    if collection:
        cfg.collection = collection

    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["store"] = MemoryStore(
        path=cfg.store_path, collection=cfg.collection, actor="cli",
    )


def _register() -> None:
    app.command("remember")(remember)
    app.command("recall")(recall)
    app.command("pack")(pack)
    app.command("auto-context")(auto_context)
    app.command("list")(list_memories)
    app.command("show")(show)
    app.command("forget")(forget)
    app.command("nuke")(nuke)
    app.command("edit")(edit)
    app.command("tag")(tag)
    # ignore_unknown_options so a negative delta like `-0.1` is taken as the
    # positional argument value instead of being parsed as an option flag.
    app.command("bump", context_settings={"ignore_unknown_options": True})(bump)
    app.command("link")(link)
    app.command("unlink")(unlink)
    app.command("neighbors")(neighbors)
    app.command("graph")(graph)
    app.command("conflicts")(conflicts)
    app.command("supersede")(supersede)
    app.command("merge")(merge)
    app.command("versions")(versions)
    app.command("path")(path)
    app.command("restore")(restore)
    app.command("prune")(prune)
    app.command("purge")(purge)
    app.command("reflect")(reflect)
    app.command("export")(export_cmd)
    app.command("import")(import_cmd)
    app.command("info")(info_cmd)
    app.command("backup")(backup)
    app.command("stats")(stats)
    app.command("metrics")(metrics)
    app.command("history")(history)
    app.command("state-at")(state_at)
    app.command("get-at")(get_at)
    app.command("doctor")(doctor)
    app.command("eval")(eval_cmd)
    app.command("ingest")(ingest)
    app.command("serve")(serve)
    app.command("tui")(tui)
    app.add_typer(maintenance_app, name="maintenance")
    app.add_typer(journal_app, name="journal")
    app.add_typer(collections_app, name="collections")
    app.add_typer(tags_app, name="tags")
    app.add_typer(trash_app, name="trash")


_register()


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()
