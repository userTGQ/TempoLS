import os
import argparse

import numpy as np
import pandas as pd
import torch

import util
from model import TempoLS


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--data", type=str, default="PEMS08_48", help="dataset name")
parser.add_argument("--input_dim", type=int, default=3, help="input dimension")
parser.add_argument("--channels", type=int, default=128, help="model channels")
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input length")
parser.add_argument("--output_len", type=int, default=12, help="prediction length")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./logs/2026-10-22-11-10-57-PEMS08/best_model.pth",
    help="model checkpoint path",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./results",
    help="directory to save CSV files",
)
parser.add_argument(
    "--continuous_horizon",
    type=int,
    default=6,
    help="1-based forecast horizon used to build continuous CSV files (default: 6)",
)
args = parser.parse_args()


def save_continuous_csv(values, csv_path, horizon_index=5):
    """Save one forecast horizon from every sliding window as a continuous series.

    values has shape [samples, nodes, horizons]. The default horizon_index=5
    selects the 6th predicted time step from every sample/window.
    """
    array = (
        values.detach().cpu().numpy()
        if torch.is_tensor(values)
        else np.asarray(values)
    )

    if array.ndim != 3:
        raise ValueError(
            f"Expected [samples, nodes, horizons], got shape {array.shape}"
        )
    if not 0 <= horizon_index < array.shape[2]:
        raise ValueError(
            f"horizon_index={horizon_index} is invalid for output length "
            f"{array.shape[2]}; expected 0 <= horizon_index < {array.shape[2]}."
        )

    # Select the same forecast horizon from EVERY sliding window.
    continuous = array[:, :, horizon_index]  # [samples, nodes]

    df = pd.DataFrame(
        continuous,
        columns=[f"node_{i}" for i in range(continuous.shape[1])],
    )
    df.insert(0, "time_idx", np.arange(len(df)))
    df.to_csv(csv_path, index=False, float_format="%.6f", encoding="utf-8-sig")
    return continuous.shape


def main():
    data_name = args.data

    if args.data == "PEMS08":
        args.data = "data//" + args.data
        args.num_nodes = 170
    elif args.data in ("PEMS08_36", "PEMS08_48"):
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.input_len = 36 if args.data.endswith("36") else 48
        args.output_len = args.input_len
    elif args.data == "PEMS03":
        args.data = "data//" + args.data
        args.num_nodes = 358
    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        args.num_nodes = 307
    elif args.data in ("PEMS04_36", "PEMS04_48"):
        args.data = "data//" + args.data
        args.num_nodes = 307
        args.input_len = 36 if args.data.endswith("36") else 48
        args.output_len = args.input_len
    elif args.data == "PEMS07":
        args.data = "data//" + args.data
        args.num_nodes = 883
    elif args.data == "bike_drop":
        args.data = "data//" + args.data
        args.num_nodes = 250
    elif args.data == "bike_pick":
        args.data = "data//" + args.data
        args.num_nodes = 250
    elif args.data == "taxi_drop":
        args.data = "data//" + args.data
        args.num_nodes = 266
    elif args.data == "taxi_pick":
        args.data = "data//" + args.data
        args.num_nodes = 266

    if not 1 <= args.continuous_horizon <= args.output_len:
        raise ValueError(
            f"--continuous_horizon must be between 1 and output_len "
            f"({args.output_len}), got {args.continuous_horizon}."
        )
    horizon_index = args.continuous_horizon - 1

    device = torch.device(args.device)

    model = TempoLS(
        device,
        args.input_dim,
        args.channels,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.dropout,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Support checkpoints saved as a state_dict or as {'state_dict': ...}.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully.")

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    outputs = []
    realy = torch.as_tensor(
        dataloader["y_test"], dtype=torch.float32, device=device
    )
    realy = realy.transpose(1, 3)[:, 0, :, :]  # [samples, nodes, output_len]

    with torch.no_grad():
        for _, (x, _) in enumerate(dataloader["test_loader"].get_iterator()):
            testx = torch.as_tensor(x, dtype=torch.float32, device=device)
            testx = testx.transpose(1, 3)
            preds = model(testx).transpose(1, 3)  # [batch, 1, nodes, output_len]
            outputs.append(preds.squeeze(1))  # [batch, nodes, output_len]

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]
    yhat = scaler.inverse_transform(yhat)

    # Evaluate all forecast horizons as before.
    amae, amape, armse, awmape = [], [], [], []
    for horizon in range(args.output_len):
        pred = yhat[:, :, horizon]
        real = realy[:, :, horizon]
        metrics = util.metric(pred, real)
        print(
            "Evaluate test data for horizon {:d}, Test MAE: {:.4f}, "
            "Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}".format(
                horizon + 1, metrics[0], metrics[1], metrics[2], metrics[3]
            )
        )
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    print(
        "Average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, "
        "Test RMSE: {:.4f}, Test WMAPE: {:.4f}".format(
            args.output_len,
            np.mean(amae),
            np.mean(amape),
            np.mean(armse),
            np.mean(awmape),
        )
    )

    # For the continuous CSVs, take the selected horizon from EVERY window.
    # Default --continuous_horizon=6 selects the 6th predicted step.
    os.makedirs(args.output_dir, exist_ok=True)
    real_csv = os.path.join(args.output_dir, f"{data_name}_real.csv")
    pred_csv = os.path.join(args.output_dir, f"{data_name}_pred.csv")

    real_shape = save_continuous_csv(realy, real_csv, horizon_index=horizon_index)
    pred_shape = save_continuous_csv(yhat, pred_csv, horizon_index=horizon_index)

    print(
        f"Continuous series uses forecast horizon {args.continuous_horizon} "
        "from every sliding window."
    )
    print(f"Continuous real series shape [time_steps, nodes]: {real_shape}")
    print(f"Continuous prediction series shape [time_steps, nodes]: {pred_shape}")
    print(f"Real values CSV saved to: {os.path.abspath(real_csv)}")
    print(f"Predictions CSV saved to: {os.path.abspath(pred_csv)}")


if __name__ == "__main__":
    main()
