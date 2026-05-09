"""houkai — AI-Houkai CLI entrypoint."""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(
    name="houkai",
    help="Manage AI-Houkai memories from the terminal.",
    no_args_is_help=True,
    add_completion=True,
)


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
) -> None:
    from ai_houkai.cli.config import load as load_config
    from ai_houkai.memory_system import MemoryStore

    cfg = load_config()
    if store_path:
        cfg.store_path = store_path
    if collection:
        cfg.collection = collection

    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["store"] = MemoryStore(path=cfg.store_path, collection=cfg.collection)


def _register() -> None:
    from ai_houkai.cli.commands.remember import remember
    from ai_houkai.cli.commands.recall import recall
    from ai_houkai.cli.commands.list_cmd import list_memories
    from ai_houkai.cli.commands.show import show
    from ai_houkai.cli.commands.forget import forget
    from ai_houkai.cli.commands.edit import edit, tag, bump
    from ai_houkai.cli.commands.link import link, unlink, neighbors, graph
    from ai_houkai.cli.commands.conflicts import conflicts, supersede, restore
    from ai_houkai.cli.commands.decay import prune
    from ai_houkai.cli.commands.reflect import reflect
    from ai_houkai.cli.commands.io import export_cmd, import_cmd, backup
    from ai_houkai.cli.commands.stats import stats
    from ai_houkai.cli.commands.maintenance import maintenance_app

    app.command("remember")(remember)
    app.command("recall")(recall)
    app.command("list")(list_memories)
    app.command("show")(show)
    app.command("forget")(forget)
    app.command("edit")(edit)
    app.command("tag")(tag)
    app.command("bump")(bump)
    app.command("link")(link)
    app.command("unlink")(unlink)
    app.command("neighbors")(neighbors)
    app.command("graph")(graph)
    app.command("conflicts")(conflicts)
    app.command("supersede")(supersede)
    app.command("restore")(restore)
    app.command("prune")(prune)
    app.command("reflect")(reflect)
    app.command("export")(export_cmd)
    app.command("import")(import_cmd)
    app.command("backup")(backup)
    app.command("stats")(stats)
    app.add_typer(maintenance_app, name="maintenance")


_register()


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()
