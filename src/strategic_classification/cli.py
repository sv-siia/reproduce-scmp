import typer
from strategic_classification.models.rnn import train_rnn
from strategic_classification.models.recourse import train_recourse
from strategic_classification.models.utility import train_utility
from strategic_classification.models.batched import train_batched
from strategic_classification.models.manifold import train_manifold

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
    elif model == "recourse":
        typer.echo("Training Recourse model...")
        # python src/strategic_classification/cli.py train recourse dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/recourse
        # scmp train recourse dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/recourse
        train_recourse(dataset_path, epochs, batch_size, model_checkpoint_path)
    elif model == "manifold":
        typer.echo("Training Maniforld model...")
        # python src/strategic_classification/cli.py train manifold dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/manifold
        # scmp train manifold dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/manifold
        train_manifold(dataset_path, epochs, batch_size, model_checkpoint_path)
    elif model == "utility":
        typer.echo("Training Utility model...")
        # python src/strategic_classification/cli.py train utility dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/utility
        # scmp train utility dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/utility
        train_utility(dataset_path, epochs, batch_size, model_checkpoint_path)
    elif model == "rac":
        typer.echo("Training RobustnessAroundCost model...")
        # TODO: Add RobustnessAroundCost model training logic here
    elif model == "batched":
        typer.echo("Training Batched model...")
        # scmp train batched dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/batched
        train_batched(epochs, batch_size, model_checkpoint_path)
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
