include("dataloader.jl")
include("model.jl")
include("train.jl")

using .TrainMNIST

# 运行训练
model = TrainMNIST.train_and_save(
    model_name="mlp",
    epochs=10,
    lr=1e-3,
    batch_size=128,
    out_path="models/map.bson"
) 