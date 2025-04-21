import typer
from strategic_classification.models.rnn import train_rnn

app = typer.Typer(help="CLI for the reproduce-scmp project.")

@app.command()
def train(model: str, dataset_path: str, epochs: int = 5, batch_size: int = 16, model_checkpoint_path: str = "models/rnn"):
    """
    Train a specified model.

    Args:
        model (str): The name of the model to train (e.g., 'rnn', 'vanilla').
    """
    if model == "rnn":
        typer.echo("Training RNN model...")
        # python src/strategic_classification/cli.py train rnn dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/rnn
        # scmp train rnn dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/rnn
        train_rnn(dataset_path, epochs, batch_size, model_checkpoint_path)
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
