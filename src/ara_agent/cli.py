"""Simple CLI launcher for ara-agent"""

import typer
from rich.console import Console
from rich.panel import Panel

from ara_agent.voice_agent import main as voice_main

app = typer.Typer(help="ara — Voice-first Grok computer agent")
console = Console()


@app.command()
def start(
    menu: bool = typer.Option(
        False, "--menu", "-m",
        help="Launch as a macOS menu bar app (with an animated blackhole icon).",
    ),
):
    """Start talking to Ara"""
    if menu:
        from ara_agent.menu_bar import run_menu_bar
        run_menu_bar()
        return

    console.print(Panel.fit(
        "[bold cyan]Ara[/bold cyan] — Voice-first Grok agent\n\n"
        "Just start speaking. She can run commands, read files, and help you.\n"
        "Press Ctrl+C to stop.",
        title="Welcome",
        border_style="cyan"
    ))
    import asyncio
    asyncio.run(voice_main())


if __name__ == "__main__":
    app()
