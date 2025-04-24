import typer
from strategic_classification.models.rnn import train_rnn
from strategic_classification.models.recourse import train_recourse
from strategic_classification.models.utility import train_utility
from strategic_classification.models.batched import train_batched
from strategic_classification.models.manifold import train_manifold
from strategic_classification.models.burden import train_burden
from strategic_classification.models.vanila import train_vanila

app = typer.Typer(help="CLI for the reproduce-scmp project.")
train_app = typer.Typer(help="Train specific models.")
app.add_typer(train_app, name="train")


@train_app.command("rnn")
def train_rnn_model(dataset_path: str, epochs: int = 5, batch_size: int = 16, model_checkpoint_path: str = "models/rnn"):
    """
    Train the RNN model.
    """
    # python src/strategic_classification/cli.py train rnn dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/rnn
    # scmp train rnn dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/rnn
    typer.echo("Training RNN model...")
    train_rnn(dataset_path, epochs, batch_size, model_checkpoint_path)


@train_app.command("recourse")
def train_recourse_model(dataset_path: str, epochs: int = 5, batch_size: int = 16, model_checkpoint_path: str = "models/recourse"):
    """
    Train the Recourse model.
    """
    # python src/strategic_classification/cli.py train recourse dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/recourse
    # scmp train recourse dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/recourse
    typer.echo("Training Recourse model...")
    train_recourse(dataset_path, epochs, batch_size, model_checkpoint_path)


@train_app.command("utility")
def train_utility_model(dataset_path: str, epochs: int = 5, batch_size: int = 16, model_checkpoint_path: str = "models/utility"):
    """
    Train the Utility model.
    """
    # python src/strategic_classification/cli.py train utility dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/utility
    # scmp train utility dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/utility
    typer.echo("Training Utility model...")
    train_utility(dataset_path, epochs, batch_size, model_checkpoint_path)


@train_app.command("batched")
def train_batched_model(epochs: int = 5, batch_size: int = 16, model_checkpoint_path: str = "models/batched"):
    """
    Train the Batched model.
    """
    # python src/strategic_classification/cli.py train batched dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/batched
    # scmp train batched dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/batched
    typer.echo("Training Batched model...")
    train_batched(epochs, batch_size, model_checkpoint_path)


@train_app.command("burden")
def train_burden_model(epochs: int = 5, batch_size: int = 16, model_checkpoint_path: str = "models/burden"):
    """
    Train the Burden model.
    """
    # python src/strategic_classification/cli.py train burden dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/burden
    # scmp train burden dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/burden
    typer.echo("Training Burden model...")
    train_burden(epochs, batch_size, model_checkpoint_path)


@train_app.command("vanila")
def train_vanila_model(epochs: int = 5, batch_size: int = 16, model_checkpoint_path: str = "models/vanila"):
    """
    Train the Vanila model.
    """
    # python src/strategic_classification/cli.py train vanila dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/vanila
    # scmp train vanila dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/vanila
    typer.echo("Training Vanila model...")
    train_vanila(epochs, batch_size, model_checkpoint_path)


@train_app.command("manifold")
def train_manifold_model(epochs: int = 5, batch_size: int = 16, model_checkpoint_path: str = "models/manifold"):
    """
    Train the Manifold model.
    """
    # python src/strategic_classification/cli.py train manifold dataset --epochs 1 --batch-size 32 --model-checkpoint-path models/manifold
    # scmp train manifold dataset --epochs 2 --batch-size 16 --model-checkpoint-path models/manifold
    typer.echo("Training Manifold model...")
    train_manifold(epochs, batch_size, model_checkpoint_path)


@app.command()
def evaluate():
    """Evaluate the trained models."""
    typer.echo("Evaluating models...")
    # TODO: Add evaluation logic here


if __name__ == "__main__":
    app()
