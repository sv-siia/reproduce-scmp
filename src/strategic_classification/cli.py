import typer

app = typer.Typer(help="CLI for the reproduce-scmp project.")

@app.command()
def train(model: str):
    """
    Train a specified model.

    Args:
        model (str): The name of the model to train (e.g., 'rnn', 'vanilla').
    """
    if model == "rnn":
        typer.echo("Training RNN model...")
        # TODO: Add RNN training logic here
    elif model == "vanilla":
        typer.echo("Training Vanilla model...")
        # TODO: Add Vanilla model training logic here
    else:
        typer.echo(f"Unknown model: {model}")

@app.command()
def evaluate():
    """Evaluate the trained models."""
    typer.echo("Evaluating models...")
    # TODO: Add evaluation logic here

@app.command()
def preprocess():
    """Preprocess the data."""
    typer.echo("Preprocessing data...")
    # TODO: Add preprocessing logic here

if __name__ == "__main__":
    app()
