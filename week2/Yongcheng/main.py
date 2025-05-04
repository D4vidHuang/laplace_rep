import argparse
import torch
import os
from datasets import DatasetLoader
from models import MLP, LeNet
from train import train_map
from evaluate import evaluate_model, compute_ood_metrics, visualize_samples
from laplace import Laplace

def main():
    parser = argparse.ArgumentParser(description="Laplace Redux Experiment")
    parser.add_argument('--model', choices=['mlp', 'lenet'], default='mlp', help="Model type")
    parser.add_argument('--mode', choices=['map', 'la', 'la_star'], default='map', help="Evaluation mode")
    parser.add_argument('--ood', choices=['emnist', 'fmnist', 'kmnist'], default='emnist', help="OOD dataset")
    parser.add_argument('--epochs', type=int, default=20, help="Training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    args = parser.parse_args()

    base_dir = './laplace_redux_exp'
    data_dir = f'{base_dir}/data'
    model_dir = f'{base_dir}/models'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = DatasetLoader(data_dir, args.batch_size)

    train_loader = data_loader.get_mnist(train=True)
    test_loader = data_loader.get_mnist(train=False)
    ood_loader = data_loader.get_ood_dataset(args.ood)

    model = MLP() if args.model == 'mlp' else LeNet()
    model_name = args.model

    map_path = os.path.join(model_dir, f'{model_name}_map.pt')
    if not os.path.exists(map_path):
        print(f"Training {model_name} MAP model...")
        model = train_map(model, train_loader, model_dir, model_name, epochs=args.epochs)

    model.load_state_dict(torch.load(map_path, map_location=device))
    model.to(device)

    if args.mode in ['la', 'la_star']:
        la_path = os.path.join(model_dir, f'{model_name}_{args.mode}.pt')
        hessian = 'kron' if args.mode == 'la' else 'full'

        if not os.path.exists(la_path):
            print(f"Applying {args.mode} to {model_name}...")
            try:
                la_model = Laplace(
                    model=model,
                    likelihood='classification',
                    subset_of_weights='last_layer',
                    hessian_structure=hessian
                )
                la_model.fit(train_loader)
                la_model.optimize_prior_precision(method='marglik')
                torch.save(la_model.state_dict(), la_path)
                print(f"Saved Laplace model to {la_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to apply Laplace ({args.mode}): {str(e)}")
        else:
            print(f"Loading existing {args.mode} model for {model_name}...")
            la_model = Laplace(
                model=model,
                likelihood='classification',
                subset_of_weights='last_layer',
                hessian_structure=hessian
            )
            la_model.load_state_dict(torch.load(la_path, map_location=device))

        eval_model = la_model
        is_laplace = True
    else:
        eval_model = model
        is_laplace = False

    id_acc, id_conf, id_scores, id_targets, id_preds, id_labels, id_images = evaluate_model(
        eval_model, test_loader, is_ood=False, is_laplace=is_laplace
    )
    ood_acc, ood_conf, ood_scores, ood_targets, _, _, _ = evaluate_model(
        eval_model, ood_loader, is_ood=True, is_laplace=is_laplace
    )

    auroc, fpr95 = compute_ood_metrics(id_scores, id_targets, ood_scores, ood_targets)

    print(f"[{args.mode.upper()}] {model_name.upper()} on MNIST (ID):")
    print(f"Accuracy: {id_acc * 100:.2f}%, Confidence: {id_conf:.4f}")
    print(f"[{args.mode.upper()}] {model_name.upper()} on {args.ood.upper()} (OOD):")
    print(f"Accuracy: {ood_acc * 100:.2f}%, Confidence: {ood_conf:.4f}")
    print(f"OOD Detection: AUROC: {auroc:.4f}, FPR@95: {fpr95:.4f}")

    visualize_samples(
        id_images[:10], id_preds[:10], id_labels[:10],
        title=f"{args.mode.upper()} - {model_name.upper()} ID Samples",
        save_path=os.path.join(base_dir, f'{model_name}_{args.mode}_id_samples.png')
    )

if __name__ == "__main__":
    main()