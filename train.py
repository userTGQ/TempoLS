import torch
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from model import TempoLS
from ranger21 import Ranger
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")

parser.add_argument("--data", type=str, default="PEMS08_48", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=128, help="number of nodes")
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument("--epochs", type=int, default=300, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "-",
    help="save path",
)
parser.add_argument(
    "--es_patience",
    type=int,
    default=100,
    help="quit if no improvement after this many iterations",
)
parser.add_argument(
    "--test_start_epoch", "--test-start-epoch",
    type=int,
    default=40,
    help="Start printing test metrics on validation improvements after this epoch (default: 40; starts at epoch 41).",
)
parser.add_argument(
    "--test_on_update", "--test-on-update",
    dest="test_on_update",
    action="store_true",
    help="Enable intermediate test evaluation on validation-improved checkpoints after test_start_epoch (default: disabled).",
)
parser.add_argument(
    "--no_test_on_update", "--no-test-on-update",
    dest="test_on_update",
    action="store_false",
    help="Disable intermediate test evaluation; the final test evaluation still runs after training (default).",
)
parser.set_defaults(test_on_update=False)
args = parser.parse_args()


def get_cuda_peak_memory_mb(device):
    """
    与参考训练脚本保持一致：
    返回本 epoch reset_peak_memory_stats() 之后的峰值 allocated/reserved 显存（MiB）。
    allocated：PyTorch 实际分配给张量的峰值显存；
    reserved：PyTorch CUDA caching allocator 保留的峰值显存。
    """
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0

    torch.cuda.synchronize(device)
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return peak_allocated_mb, peak_reserved_mb


def report_model_complexity(model, device, input_dim, num_nodes, input_len, batch_size):
    """Print trainable parameters and estimated FLOPs for the configured batch size."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.4f} M)")

    dummy = torch.zeros(batch_size, input_dim, num_nodes, input_len, dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    try:
        try:
            from thop import profile as thop_profile
            with torch.no_grad():
                macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            print(f"FLOPs: {2.0 * float(macs) / 1e9:.4f} GFLOPs (batch size = {batch_size}; 1 MAC = 2 FLOPs)")
        except Exception:
            try:
                from fvcore.nn import FlopCountAnalysis
                with torch.no_grad():
                    flops = float(FlopCountAnalysis(model, (dummy,)).total())
                print(f"FLOPs: {flops / 1e9:.4f} GFLOPs (batch size = {batch_size})")
            except Exception:
                print("FLOPs: unable to estimate; install thop with `pip install thop`.")
    finally:
        model.train(was_training)

    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        print("Peak GPU memory: unavailable (CUDA is not active)")


def evaluate_test_on_update(model, dataloader, scaler, device, output_len, epoch):
    """Print test metrics for monitoring only; never use test metrics to select checkpoints."""
    was_training = model.training
    model.eval()
    outputs = []
    try:
        realy = torch.as_tensor(dataloader["y_test"], dtype=torch.float32, device=device)
        realy = realy.transpose(1, 3)[:, 0, :, :]
        with torch.no_grad():
            for _, (x, _) in enumerate(dataloader["test_loader"].get_iterator()):
                testx = torch.as_tensor(x, dtype=torch.float32, device=device).transpose(1, 3)
                # Keep the batch dimension even if the last batch has only one sample.
                preds = model(testx).transpose(1, 3).squeeze(1)
                outputs.append(preds)
        if not outputs:
            print(f"[Test][Epoch {epoch}] Test loader is empty; skipping intermediate test output.")
            return
        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        amae, amape, armse, awmape = [], [], [], []
        print(f"[Test][Epoch {epoch}] Validation improved; test metrics (monitoring only):")
        for horizon in range(output_len):
            pred = scaler.inverse_transform(yhat[:, :, horizon])
            real = realy[:, :, horizon]
            metrics = util.metric(pred, real)
            print(
                "  Horizon {:d}: Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
                    horizon + 1, metrics[0], metrics[2], metrics[1], metrics[3]
                ),
                flush=True,
            )
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
            awmape.append(metrics[3])

        print(
            "[Test][Epoch {}] Average over {} horizons: MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, WMAPE: {:.4f}".format(
                epoch, output_len, np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)
            ),
            flush=True,
        )
    finally:
        model.train(was_training)


class trainer:
    def __init__(
            self,
            scaler,
            input_dim,
            channels,
            num_nodes,
            input_len,
            output_len,
            dropout,
            lrate,
            wdecay,
            device,
    ):
        self.model = TempoLS(
            input_dim=input_dim,
            num_nodes=num_nodes,
            input_len=input_len,
            output_len=output_len,
            dropout=dropout,
        )
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5
        print(self.model)
        # exit()

    def train(self, input, real_val):
        # input: [B, input_dim, N, T]
        # real_val: [B, N, output_len]
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)  # 64 12 170 1
        output = output.transpose(1, 3)  # 64 1 170 12
        real = torch.unsqueeze(real_val, dim=1)  # 64 1 170 12
        predict = self.scaler.inverse_transform(output)  # 64 1 170 12
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
            output = output.transpose(1, 3)
            real = torch.unsqueeze(real_val, dim=1)
            predict = self.scaler.inverse_transform(output)
            loss = self.loss(predict, real, 0.0)
            mape = util.MAPE_torch(predict, real, 0.0).item()
            rmse = util.RMSE_torch(predict, real, 0.0).item()
            wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def main():
    seed_it(6666)

    data = args.data

    if args.data == "PEMS08":
        args.data = "data//" + args.data
        args.num_nodes = 170
    elif args.data == "PEMS03_36":
        args.data = "data//" + args.data
        args.num_nodes = 358
        args.input_len = 36
        args.output_len = 36
    elif args.data == "PEMS03_24":
        args.data = "data//" + args.data
        args.num_nodes = 358
        args.input_len = 24
        args.output_len = 24
    elif args.data == "PEMS03_48":
        args.data = "data//" + args.data
        args.num_nodes = 358
        args.input_len = 48
        args.output_len = 48
    elif args.data == "PEMS08_36":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.input_len = 36
        args.output_len = 36
    elif args.data == "PEMS08_24":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.input_len = 24
        args.output_len = 24
    elif args.data == "PEMS08_48":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.input_len = 48
        args.output_len = 48

    elif args.data == "PEMS08_60":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.input_len = 60
        args.output_len = 60
    elif args.data == "PEMS04_60":
        args.data = "data//" + args.data
        args.num_nodes = 307
        args.input_len = 60
        args.output_len = 60
    elif args.data == "PEMS04_36":
        args.data = "data/PEMS04_36"
        args.num_nodes = 307
        args.input_len = 36
        args.output_len = 36
    elif args.data == "PEMS04_24":
        args.data = "data/PEMS04_24"
        args.num_nodes = 307
        args.input_len = 24
        args.output_len = 24
    elif args.data == "PEMS04_48":
        args.data = "data/PEMS04_48"
        args.num_nodes = 307
        args.input_len = 48
        args.output_len = 48
    elif args.data == "PEMS07_24":
        args.data = "data/PEMS07_24"
        args.num_nodes = 883
        args.input_len = 24
        args.output_len = 24
    elif args.data == "SEATTLE_36":
        args.data = "data/SEATTLE_36"
        args.num_nodes = 323
        args.input_len = 24
        args.output_len = 24
    elif args.data == "SEATTLE_48":
        args.data = "data/SEATTLE_48"
        args.num_nodes = 323
        args.input_len = 48
        args.output_len = 48
    elif args.data == "SEATTLE_36":
        args.data = "data/SEATTLE_36"
        args.num_nodes = 323
        args.input_len = 36
        args.output_len = 36

    elif args.data == "PEMS03":
        args.data = "data//" + args.data
        args.num_nodes = 358
        args.epochs = 300
        args.es_patience = 100

    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        args.num_nodes = 307

    elif args.data == "PEMS07":
        args.data = "data/PEMS07"
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

    device = torch.device(args.device)

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    loss = 9999999
    test_log = 999999
    epochs_since_best_mae = 0
    lr_no_improve_epochs = 0
    path = args.save + data + "/"

    his_loss = []
    val_time = []
    train_time = []
    train_peak_allocated_list = []
    train_peak_reserved_list = []
    result = []
    test_result = []

    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = trainer(
        scaler,
        args.input_dim,
        args.channels,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
    )

    report_model_complexity(
        engine.model,
        device,
        input_dim=args.input_dim,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        batch_size=args.batch_size,
    )

    print("start training...", flush=True)

    for i in range(1, args.epochs + 1):
        # train
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        # 与参考代码一致：每个 epoch 开始时重置峰值显存统计。
        # 后续峰值覆盖本 epoch 的全部 batch（forward + backward + optimizer）。
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        t1 = time.time()
        engine.model.train()
        # dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device)  # 64 12 170 1
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

            if iter % args.print_every == 0:
                log = "Iter: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}"
                print(
                    log.format(
                        iter,
                        train_loss[-1],
                        train_rmse[-1],
                        train_mape[-1],
                        train_wmape[-1],
                    ),
                    flush=True,
                )

        # 等待本 epoch 的 CUDA 运算完成，再读取整个训练 epoch 的峰值。
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        peak_allocated_mb, peak_reserved_mb = get_cuda_peak_memory_mb(device)
        train_peak_allocated_list.append(peak_allocated_mb)
        train_peak_reserved_list.append(peak_reserved_mb)

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        print(
            "Epoch: {:03d}, Train Peak GPU: {:.2f} MiB ({:.3f} GiB), "
            "Peak Reserved: {:.2f} MiB ({:.3f} GiB)".format(
                i,
                peak_allocated_mb,
                peak_allocated_mb / 1024,
                peak_reserved_mb,
                peak_reserved_mb / 1024,
            ),
            flush=True,
        )
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

        s2 = time.time()

        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        train_m = dict(
            train_loss=np.mean(train_loss),
            train_rmse=np.mean(train_rmse),
            train_mape=np.mean(train_mape),
            train_wmape=np.mean(train_wmape),
            valid_loss=np.mean(valid_loss),
            valid_rmse=np.mean(valid_rmse),
            valid_mape=np.mean(valid_mape),
            valid_wmape=np.mean(valid_wmape),
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, "
        print(
            log.format(i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape),
            flush=True,
        )

        if mvalid_loss < loss:
            print("### Validation loss improved; updating best checkpoint ###")
            loss = mvalid_loss
            torch.save(engine.model.state_dict(), path + "best_model.pth")
            bestid = i
            epochs_since_best_mae = 0
            print("Updating! Valid Loss:", mvalid_loss, end=", ")
            print("epoch: ", i)

            # By default, after epoch 40, print test metrics whenever validation improves.
            # The test set is for monitoring only; it is not used to choose the checkpoint.
            if args.test_on_update and i > args.test_start_epoch:
                evaluate_test_on_update(
                    engine.model,
                    dataloader,
                    scaler,
                    device,
                    args.output_len,
                    epoch=i,
                )
        else:
            epochs_since_best_mae += 1
            print("No update")



        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")
        if epochs_since_best_mae >= args.es_patience and i >= 300:
            break

    # 整个训练过程中，各 epoch 训练峰值的最大值（与参考脚本一致）。
    if train_peak_allocated_list:
        max_train_peak_allocated_mb = max(train_peak_allocated_list)
        max_train_peak_reserved_mb = max(train_peak_reserved_list)
        print(
            "Max Train Peak GPU Memory = {:.2f} MiB ({:.3f} GiB)".format(
                max_train_peak_allocated_mb,
                max_train_peak_allocated_mb / 1024,
            ),
            flush=True,
        )
        print(
            "Max Train Peak Reserved Memory = {:.2f} MiB ({:.3f} GiB)".format(
                max_train_peak_reserved_mb,
                max_train_peak_reserved_mb / 1024,
            ),
            flush=True,
        )

    # Output consumption
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # test
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    engine.model.load_state_dict(torch.load(path + "best_model.pth"))
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    awmape = []

    test_m = []

    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[2], metrics[1], metrics[3]))

        test_m = dict(
            test_loss=np.mean(metrics[0]),
            test_rmse=np.mean(metrics[2]),
            test_mape=np.mean(metrics[1]),
            test_wmape=np.mean(metrics[3]),
        )
        test_m = pd.Series(test_m)
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = "On average over {} horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(args.output_len, np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)))

    test_m = dict(
        test_loss=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean(awmape),
    )
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{path}/test.csv")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))

