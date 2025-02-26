import click  # type: ignore
import json
from rich.console import Console  # type: ignore
from rich.table import Table  # type: ignore
from rich.syntax import Syntax  # type: ignore
from rich.panel import Panel  # type: ignore
from kernel_agent.kernel_optimizer import HardwareOptimizationAgent  # type:ignore

console = Console()


def display_hardware_analysis(hardware_specs: dict):
    table = Table(title="Hardware Analysis")
    table.add_column("Aspect", style="cyan")
    table.add_column("Details", style="green")

    for key, value in hardware_specs.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


def display_performance_metrics(metrics: dict):
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in metrics.items():
        formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
        table.add_row(key.replace("_", " ").title(), formatted_value)

    console.print(table)


def display_code(code: str):
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Generated Code", border_style="green"))


@click.group()
def cli():
    """Hardware Optimization Agent CLI"""
    pass


@cli.command()
@click.argument("hardware_description", type=str)
@click.option("--model", "-m", default="gpt-4-turbo-preview", help="LLM model to use")
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
def optimize(hardware_description: str, model: str, output: str, verbose: bool):
    """Optimize code for given hardware description"""

    with console.status("[bold green]Initializing agent..."):
        agent = HardwareOptimizationAgent(model_name=model)

    console.print(
        f"\n🚀 Starting optimization for hardware:\n[yellow]{hardware_description}[/yellow]\n"
    )

    with console.status("[bold green]Running optimization workflow..."):
        result = agent.optimize(hardware_description)

    console.print("\n✨ Optimization complete!\n")

    # Display results
    if result["hardware_specs"]:
        display_hardware_analysis(result["hardware_specs"])
        console.print()

    if result["code"]:
        display_code(result["code"])
        console.print()

    if result["performance_metrics"]:
        display_performance_metrics(result["performance_metrics"])
        console.print()

    console.print(
        f"\nSatisfaction Level: [{'green' if result['is_satisfied'] else 'red'}]{result['satisfaction_level']:.2f}[/]"
    )

    if output:
        with open(output, "w") as f:
            json.dump(
                {
                    "code": result["code"],
                    "performance_metrics": result["performance_metrics"],
                    "assessment": result["assessment"],
                    "satisfaction_level": result["satisfaction_level"],
                },
                f,
                indent=2,
            )
        console.print(f"\nResults saved to: [blue]{output}[/]")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def analyze(input_file: str):
    """Analyze previously generated optimization results"""
    with open(input_file) as f:
        data = json.load(f)

    console.print("\n📊 Analysis Results\n")

    if "code" in data:
        display_code(data["code"])
        console.print()

    if "performance_metrics" in data:
        display_performance_metrics(data["performance_metrics"])
        console.print()

    if "satisfaction_level" in data:
        console.print(
            f"\nSatisfaction Level: [{'green' if data['satisfaction_level'] >= 7.0 else 'red'}]{data['satisfaction_level']:.2f}[/]"
        )


@cli.command()
def models():
    """List available LLM models"""
    models = [
        ("gpt-4-turbo-preview", "Latest GPT-4 model with best performance"),
        ("gpt-3.5-turbo", "Faster, more economical option"),
        ("gpt-4", "Standard GPT-4 model"),
    ]

    table = Table(title="Available Models")
    table.add_column("Model", style="cyan")
    table.add_column("Description", style="green")

    for model, desc in models:
        table.add_row(model, desc)

    console.print(table)


if __name__ == "__main__":
    cli()
